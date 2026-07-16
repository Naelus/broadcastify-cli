using Windows.Security.Credentials;

namespace BroadcastifyCli.WinUI;

internal sealed record SavedLogin(string Username, string Password);

internal static class CredentialStore
{
    private const string Resource = "BroadcastifyDesktop.Broadcastify";

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
}
