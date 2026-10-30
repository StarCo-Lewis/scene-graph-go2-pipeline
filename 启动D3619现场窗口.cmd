@echo off
setlocal
cd /d "%~dp0"
if defined FIELD_UI_PYTHON goto run
set "FIELD_UI_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%FIELD_UI_PYTHON%" goto run
set "FIELD_UI_PYTHON=python"
:run
"%FIELD_UI_PYTHON%" -B -m embodied_pipeline field-console
if errorlevel 1 pause
endlocal
