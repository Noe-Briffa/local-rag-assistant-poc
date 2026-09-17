@echo off
cd /d "%~dp0"
REM (Optionnel) active le venv
IF EXIST .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

python start_llama.py
pause
