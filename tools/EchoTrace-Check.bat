@echo off
rem EchoTrace PC check. Double-click to run. No administrator rights needed.
rem It only reads information, speaks a short message, and writes a report to the Desktop.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0EchoTrace-Check.ps1"
echo.
echo Press any key to close this window.
pause >nul
