@echo off
rem One short training sitting for the personal tap model (~3.5 min). Run once per day or so,
rem sitting a little differently each time. Then: .venv\Scripts\python -m handctl.evaluate_taps
cd /d "%~dp0"
".venv\Scripts\python.exe" -m handctl.record --protocol session --position "%*"
pause
