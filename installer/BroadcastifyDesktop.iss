#ifndef MyAppVersion
  #define MyAppVersion "0.4.63"
#endif
#ifndef MyAppVersionNumeric
  #define MyAppVersionNumeric "0.4.63.0"
#endif
#ifndef SourceDir
  #error SourceDir must point to the prepared Windows application directory.
#endif
#ifndef OutputDir
  #error OutputDir must point to the installer output directory.
#endif

#define MyAppName "Broadcastify Desktop"
#define MyAppExeName "Broadcastify Desktop.exe"
#define MyAppPublisher "Radio Archive Project"
#define MyAppUrl "https://github.com/Naelus/broadcastify-cli"

[Setup]
AppId={{3B7D8D50-659D-4C72-9A84-0DA39472F8A3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppUrl}
AppSupportURL={#MyAppUrl}/issues
AppUpdatesURL={#MyAppUrl}/releases
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile={#SourceDir}\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=BroadcastifyDesktop-{#MyAppVersion}-win-x64-setup
SetupIconFile={#SourceDir}\BroadcastifyDesktop.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
CloseApplicationsFilter={#MyAppExeName}
RestartApplications=no
UsePreviousAppDir=yes
UsePreviousGroup=yes
UsePreviousTasks=yes
SetupLogging=yes
VersionInfoVersion={#MyAppVersionNumeric}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoTextVersion={#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Rebuild managed runtime trees on every upgrade so removed packages and
; version metadata cannot survive an otherwise successful in-place install.
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\windowsml"
; Public releases no longer contain symbols. Remove a root symbol left by an
; earlier developer or private build during an in-place upgrade.
Type: files; Name: "{app}\*.pdb"
#if !FileExists(SourceDir + "\broadcastify-desktop.env")
; A clean/public upgrade must scrub an owner-only environment file that may
; have been installed by an earlier private build.
Type: files; Name: "{app}\broadcastify-desktop.env"
#endif

[Dirs]
Name: "{localappdata}\Broadcastify Desktop"; Flags: uninsneveruninstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{localappdata}\Broadcastify Desktop"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{localappdata}\Broadcastify Desktop"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"; Flags: uninsdeletekey

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--post-install --prompt-setup"; Description: "Launch {#MyAppName}"; WorkingDir: "{localappdata}\Broadcastify Desktop"; Flags: nowait postinstall skipifsilent
Filename: "{app}\{#MyAppExeName}"; Parameters: "--post-install --prompt-setup"; WorkingDir: "{localappdata}\Broadcastify Desktop"; Flags: nowait; Check: ShouldLaunchAfterSilentInstall

[Code]
const
  StartupRunKey = 'Software\Microsoft\Windows\CurrentVersion\Run';
  StartupRunValue = 'Broadcastify Desktop';
  StartupPreferenceKey = 'Software\Radio Archive Project\Broadcastify Desktop';
  StartupPreferenceValue = 'StartWithWindows';

var
  StartupPage: TInputOptionWizardPage;

function HasCommandLineFlag(const FlagName: String): Boolean;
var
  Index: Integer;
  Value: String;
begin
  Result := False;
  for Index := 1 to ParamCount do
  begin
    Value := Uppercase(ParamStr(Index));
    if (Value = '/' + Uppercase(FlagName)) or
       (Value = '-' + Uppercase(FlagName)) then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

procedure InitializeWizard;
var
  SavedPreference: Cardinal;
begin
  StartupPage := CreateInputOptionPage(
    wpSelectDir,
    'Keep scheduled feeds current',
    'Start Broadcastify Desktop with Windows',
    'This visible, per-user option lets saved schedules run after sign-in, ' +
      'observe the rolling archive limit, and resume retained work after an interruption.',
    False,
    False);
  StartupPage.Add('Start Broadcastify Desktop when I sign in (recommended)');
  if RegQueryDWordValue(
       HKCU,
       StartupPreferenceKey,
       StartupPreferenceValue,
       SavedPreference) then
    StartupPage.Values[0] := SavedPreference <> 0
  else
    StartupPage.Values[0] := True;
end;

procedure SaveStartupPreference(const Enabled: Boolean);
var
  Command: String;
begin
  if Enabled then
  begin
    Command :=
      '"' + ExpandConstant('{app}\{#MyAppExeName}') +
      '" --startup --prompt-setup';
    if not RegWriteStringValue(
             HKCU,
             StartupRunKey,
             StartupRunValue,
             Command) then
      RaiseException('Windows could not enable start with sign-in.');
  end
  else
    RegDeleteValue(HKCU, StartupRunKey, StartupRunValue);

  if not RegWriteDWordValue(
           HKCU,
           StartupPreferenceKey,
           StartupPreferenceValue,
           Ord(Enabled)) then
    RaiseException('Windows could not save the startup preference.');
end;

procedure CurStepChanged(const CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;

  if WizardSilent then
  begin
    if HasCommandLineFlag('ENABLESTARTUP') then
      SaveStartupPreference(True)
    else if HasCommandLineFlag('DISABLESTARTUP') then
      SaveStartupPreference(False);
    { With neither flag, a silent install preserves an existing preference
      and leaves startup disabled on a fresh install. }
  end
  else
    SaveStartupPreference(StartupPage.Values[0]);
end;

function ShouldLaunchAfterSilentInstall: Boolean;
begin
  Result :=
    WizardSilent and HasCommandLineFlag('LAUNCHAFTERINSTALL');
end;

procedure CurUninstallStepChanged(const CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    RegDeleteValue(HKCU, StartupRunKey, StartupRunValue);
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
