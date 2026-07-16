using Windows.Security.Credentials;

namespace BroadcastifyCli.WinUI;

internal sealed record SavedLogin(string Username, string Password);

internal sealed record SavedSecret(string Secret);

internal static class CredentialStore
{
    private const string Resource = "BroadcastifyDesktop.Broadcastify";
    private const string AnalysisResource = "BroadcastifyDesktop.AnalysisProvider";

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
