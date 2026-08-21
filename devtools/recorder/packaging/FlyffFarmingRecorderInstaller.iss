#define MyAppName "FlyFF Farming Session Recorder"
#define MyAppVersion "1.9.0"
#define MyAppPublisher "Riddims"

#ifndef AppExe
  #define AppExe "dist\farming_recorder\FlyffFarmingRecorder.exe"
#endif
#ifndef ReadmeFile
  #define ReadmeFile "README.txt"
#endif
#ifndef ConfigFile
  #define ConfigFile "..\recorder_config.json"
#endif
#ifndef InstallerOutputDir
  #define InstallerOutputDir "dist\farming_recorder_installer"
#endif

[Setup]
AppId={{A0C97C4E-81F0-44A0-8787-5A43D0C9305F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\FlyffFarmingRecorder
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#InstallerOutputDir}
OutputBaseFilename=FlyffFarmingRecorderSetup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x86compatible
Uninstallable=yes
CreateUninstallRegKey=yes
UninstallDisplayName=Uninstall {#MyAppName}
UninstallDisplayIcon={app}\FlyffFarmingRecorder.exe
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ReadmeFile}"; DestDir: "{app}"; DestName: "READ_ME_FIRST.txt"; Flags: ignoreversion
Source: "{#ConfigFile}"; DestDir: "{app}"; Flags: onlyifdoesntexist

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\FlyffFarmingRecorder.exe"; WorkingDir: "{app}"
Name: "{autoprograms}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\FlyffFarmingRecorder.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\FlyffFarmingRecorder.exe"; Description: "Run the recorder now"; Flags: nowait postinstall skipifsilent

