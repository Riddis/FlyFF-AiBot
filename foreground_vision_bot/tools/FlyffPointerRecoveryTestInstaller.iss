#define MyAppName "FlyFF Pointer Recovery Test"
#define MyAppVersion "1.0"
#define MyAppPublisher "Riddims"

#ifndef AppExe
  #define AppExe "..\dist\friend_pointer_recovery_test\FlyffPointerRecoveryTest.exe"
#endif
#ifndef ReadmeFile
  #define ReadmeFile "..\FRIEND_POINTER_RECOVERY_TEST_README.txt"
#endif
#ifndef InstallerOutputDir
  #define InstallerOutputDir "..\dist\friend_pointer_installer"
#endif

[Setup]
AppId={{E2614DB2-E10A-4E4A-A5CA-0AF89B9E485B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\FlyffPointerRecoveryTest
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#InstallerOutputDir}
OutputBaseFilename=FlyffPointerRecoveryTestSetup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x86compatible
Uninstallable=yes
CreateUninstallRegKey=yes
UninstallDisplayName=Uninstall {#MyAppName}
UninstallDisplayIcon={app}\FlyffPointerRecoveryTest.exe
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ReadmeFile}"; DestDir: "{app}"; DestName: "READ_ME_FIRST.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\FlyffPointerRecoveryTest.exe"; WorkingDir: "{app}"
Name: "{autoprograms}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\FlyffPointerRecoveryTest.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\FlyffPointerRecoveryTest.exe"; Description: "Run the pointer recovery test now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{userdocs}\FlyffPointerRecoveryTest"
