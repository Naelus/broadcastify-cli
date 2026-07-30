#ifndef MyAppVersion
  #define MyAppVersion "0.4.2"
#endif
#ifndef MyAppVersionNumeric
  #define MyAppVersionNumeric "0.4.2.0"
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
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{localappdata}\Broadcastify Desktop"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
