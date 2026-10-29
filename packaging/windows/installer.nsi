; LLMZIP NSIS Windows Installer Script
Unicode True
!define PRODUCT_NAME "LLMZIP"
!define PRODUCT_VERSION "0.1.0"
!define PRODUCT_PUBLISHER "kaohaoyi"
!define PRODUCT_WEB_SITE "https://github.com/kaohaoyi/LLMZIP"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\llmzip.exe"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

SetCompressor /SOLID lzma

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "llmzip-v0.1.0-windows-x64-setup.exe"
InstallDir "$LOCALAPPDATA\Programs\LLMZIP"
InstallDirRegKey HKCU "${PRODUCT_DIR_REGKEY}" ""
ShowInstDetails show
ShowUnInstDetails show
RequestExecutionLevel user

Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

Section "MainSection" SEC01
  SetOutPath "$INSTDIR"
  SetOverwrite ifnewer
  File "dist_staging\llmzip.exe"
  
  ; Write uninstaller
  WriteUninstaller "$INSTDIR\uninst.exe"
  
  ; Register App Path
  WriteRegStr HKCU "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\llmzip.exe"
  
  ; Register in Add/Remove Programs
  WriteRegStr HKCU "${PRODUCT_UNINST_KEY}" "DisplayName" "$(^Name)"
  WriteRegStr HKCU "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninst.exe"
  WriteRegStr HKCU "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\llmzip.exe"
  WriteRegStr HKCU "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKCU "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegStr HKCU "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"

  ; Add to User PATH
  EnVar::SetHKCU
  EnVar::AddValue "Path" "$INSTDIR"
SectionEnd

Section Uninstall
  ; Remove from User PATH
  EnVar::SetHKCU
  EnVar::DeleteValue "Path" "$INSTDIR"

  Delete "$INSTDIR\uninst.exe"
  Delete "$INSTDIR\llmzip.exe"
  RMDir "$INSTDIR"

  DeleteRegKey HKCU "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKCU "${PRODUCT_DIR_REGKEY}"
  SetAutoClose true
SectionEnd
