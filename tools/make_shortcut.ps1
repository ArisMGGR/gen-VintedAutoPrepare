# Creates VintedAutoPrepare.lnk in the project folder (called by Install.bat).
# Shortcuts store absolute paths, so this runs on your PC rather than being committed.
$root = Split-Path -Parent $PSScriptRoot
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $root 'VintedAutoPrepare.lnk'))
$shortcut.TargetPath = Join-Path $root '.venv\Scripts\pythonw.exe'  # pythonw: no console window
$shortcut.Arguments = '"' + (Join-Path $root 'run.py') + '"'
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = Join-Path $root 'assets\icon.ico'
$shortcut.Description = 'Prepare photos and listings for Vinted'
$shortcut.Save()
