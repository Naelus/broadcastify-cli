using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Windows.Security.Credentials;

namespace BroadcastifyCli.WinUI;

internal sealed record SavedLogin(string Username, string Password);

internal sealed record SavedLoginProfile(
    string Id,
    string Label,
    string Username,
    string Password);

internal sealed record BroadcastifyProfilePayload(
    string Label,
    string Username,
    string Password);

internal sealed record SavedSecret(string Secret);

internal static class CredentialStore
{
    private static string Resource =>
        ScopedResource("BroadcastifyDesktop.Broadcastify");
    private static string ProfileResource =>
        ScopedResource("BroadcastifyDesktop.BroadcastifyProfiles");
    private static string AnalysisResource =>
        ScopedResource("BroadcastifyDesktop.AnalysisProvider");
    private static string HuggingFaceResource =>
        ScopedResource("BroadcastifyDesktop.HuggingFace");

    private static string ScopedResource(string resource)
    {
        var testRoot = Environment.GetEnvironmentVariable(
            AppSettingsStore.TestDataRootEnvironment)?.Trim();
        if (string.IsNullOrWhiteSpace(testRoot))
        {
            return resource;
        }

        var identity = Path.GetFullPath(testRoot).ToUpperInvariant();
        var digest = Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(identity)));
        return $"{resource}.Test.{digest[..16]}";
    }

    public static SavedLogin? TryLoad()
    {
        var vault = new PasswordVault();
        IReadOnlyList<PasswordCredential> credentials;
        try
        {
            credentials = vault.FindAllByResource(Resource);
        }
        catch
        {
            return null;
        }
        var credential = credentials.FirstOrDefault();
        if (credential is null)
        {
            return null;
        }
        credential.RetrievePassword();
        return new SavedLogin(credential.UserName, credential.Password);
    }

    public static void Save(string username, string password)
    {
        Clear();
        new PasswordVault().Add(new PasswordCredential(Resource, username, password));
    }

    public static void Clear()
    {
        var vault = new PasswordVault();
        try
        {
            foreach (var credential in vault.FindAllByResource(Resource))
            {
                vault.Remove(credential);
            }
        }
        catch
        {
            // FindAllByResource throws when no matching credentials exist.
        }
    }

    public static IReadOnlyList<SavedLoginProfile> ListBroadcastifyProfiles()
    {
        var profiles = new List<SavedLoginProfile>();
        var primary = TryLoad();
        if (primary is not null)
        {
            profiles.Add(new SavedLoginProfile(
                "default",
                "Primary account",
                primary.Username,
                primary.Password));
        }
        var vault = new PasswordVault();
        IReadOnlyList<PasswordCredential> credentials;
        try
        {
            credentials = vault.FindAllByResource(ProfileResource);
        }
        catch
        {
            return profiles;
        }
        foreach (var credential in credentials)
        {
            try
            {
                credential.RetrievePassword();
                var payload = JsonSerializer.Deserialize<BroadcastifyProfilePayload>(
                    credential.Password);
                var profileId = NormalizeProfileId(credential.UserName);
                if (profileId == "default"
                    || payload is null
                    || string.IsNullOrWhiteSpace(payload.Username)
                    || string.IsNullOrEmpty(payload.Password))
                {
                    continue;
                }
                profiles.Add(new SavedLoginProfile(
                    profileId,
                    string.IsNullOrWhiteSpace(payload.Label)
                        ? profileId
                        : payload.Label.Trim(),
                    payload.Username.Trim(),
                    payload.Password));
            }
            catch
            {
                // Ignore one malformed legacy entry without hiding valid profiles.
            }
        }
        return profiles
            .OrderBy(value => value.Id == "default" ? 0 : 1)
            .ThenBy(value => value.Label, StringComparer.CurrentCultureIgnoreCase)
            .ToList();
    }

    public static SavedLoginProfile? TryLoadBroadcastifyProfile(string profileId)
    {
        var normalized = NormalizeProfileId(profileId);
        return ListBroadcastifyProfiles()
            .FirstOrDefault(value => value.Id == normalized);
    }

    public static void SaveBroadcastifyProfile(
        string profileId,
        string label,
        string username,
        string password)
    {
        var normalized = NormalizeProfileId(profileId);
        if (normalized == "default")
        {
            Save(username, password);
            return;
        }
        ClearBroadcastifyProfile(normalized);
        var payload = JsonSerializer.Serialize(new BroadcastifyProfilePayload(
            string.IsNullOrWhiteSpace(label) ? normalized : label.Trim(),
            username.Trim(),
            password));
        new PasswordVault().Add(new PasswordCredential(
            ProfileResource,
            normalized,
            payload));
    }

    public static void ClearBroadcastifyProfile(string profileId)
    {
        var normalized = NormalizeProfileId(profileId);
        if (normalized == "default")
        {
            Clear();
            return;
        }
        var vault = new PasswordVault();
        try
        {
            foreach (var credential in vault.FindAllByResource(ProfileResource))
            {
                if (credential.UserName.Equals(
                        normalized,
                        StringComparison.OrdinalIgnoreCase))
                {
                    vault.Remove(credential);
                }
            }
        }
        catch
        {
            // FindAllByResource throws when no matching credentials exist.
        }
    }

    public static string NormalizeProfileId(string? profileId)
    {
        var normalized = string.IsNullOrWhiteSpace(profileId)
            ? "default"
            : profileId.Trim().ToLowerInvariant();
        if (normalized.Length > 64
            || !char.IsLetterOrDigit(normalized[0])
            || normalized.Any(value =>
                !char.IsLetterOrDigit(value) && value is not '_' and not '-'))
        {
            throw new ArgumentException(
                "Account profile IDs may contain letters, numbers, underscores, and hyphens.",
                nameof(profileId));
        }
        return normalized;
    }

    public static string CreateProfileId(string label)
    {
        var builder = new StringBuilder();
        foreach (var value in (label ?? "").Trim().ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(value))
            {
                builder.Append(value);
            }
            else if (builder.Length > 0 && builder[^1] != '-')
            {
                builder.Append('-');
            }
            if (builder.Length >= 48)
            {
                break;
            }
        }
        var candidate = builder.ToString().Trim('-');
        return NormalizeProfileId(string.IsNullOrWhiteSpace(candidate)
            ? "account"
            : candidate);
    }

    public static SavedSecret? TryLoadAnalysisKey()
    {
        var credential = TryLoadCredential(AnalysisResource);
        return credential is null ? null : new SavedSecret(credential.Password);
    }

    public static void SaveAnalysisKey(string secret)
    {
        ClearResource(AnalysisResource);
        new PasswordVault().Add(new PasswordCredential(AnalysisResource, "active-key", secret));
    }

    public static void ClearAnalysisKey() => ClearResource(AnalysisResource);

    public static SavedSecret? TryLoadHuggingFaceToken()
    {
        var credential = TryLoadCredential(HuggingFaceResource);
        return credential is null ? null : new SavedSecret(credential.Password);
    }

    public static void SaveHuggingFaceToken(string secret)
    {
        ClearResource(HuggingFaceResource);
        new PasswordVault().Add(new PasswordCredential(
            HuggingFaceResource,
            "active-read-token",
            secret));
    }

    public static void ClearHuggingFaceToken() =>
        ClearResource(HuggingFaceResource);

    public static string CreateSecretPreview(
        string secret,
        int prefixLength)
    {
        if (string.IsNullOrEmpty(secret))
        {
            return "";
        }
        var visible = Math.Min(
            Math.Max(1, prefixLength),
            Math.Max(1, secret.Length - 1));
        return $"{secret[..visible]}••••";
    }

    private static PasswordCredential? TryLoadCredential(string resource)
    {
        var vault = new PasswordVault();
        IReadOnlyList<PasswordCredential> credentials;
        try
        {
            credentials = vault.FindAllByResource(resource);
        }
        catch
        {
            return null;
        }
        var credential = credentials.FirstOrDefault();
        if (credential is null)
        {
            return null;
        }
        credential.RetrievePassword();
        return credential;
    }

    private static void ClearResource(string resource)
    {
        var vault = new PasswordVault();
        try
        {
            foreach (var credential in vault.FindAllByResource(resource))
            {
                vault.Remove(credential);
            }
        }
        catch
        {
            // FindAllByResource throws when no matching credentials exist.
        }
    }
}
