@echo off
rem Personal hand scan (~75 s), then calibrate the hand model from it.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m handctl.record --protocol scan --name scan-%DATE:~-4%%DATE:~4,2%%DATE:~7,2%
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-d data\recordings\scan-*') do (
  ".venv\Scripts\python.exe" -m handctl.handmodel.calibrate "data\recordings\%%d"
  goto :end
)
:end
pause
