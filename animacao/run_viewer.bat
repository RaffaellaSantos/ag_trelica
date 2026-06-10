@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv nao encontrado. Execute setup.bat primeiro.
    pause & exit /b 1
)

echo Abrindo trelica_3d.html em http://localhost:8080 ...
echo Pressione Ctrl+C para parar o servidor.
echo.
.venv\Scripts\python.exe serve.py
pause
