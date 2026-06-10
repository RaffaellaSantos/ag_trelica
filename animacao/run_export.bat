@echo off
cd /d "%~dp0"
echo ============================================================
echo  Rodando AG e exportando dados (pode demorar alguns minutos)
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo .venv nao encontrado. Execute setup.bat primeiro.
    pause & exit /b 1
)

.venv\Scripts\python.exe export_ag_data.py
if errorlevel 1 (
    echo.
    echo ERRO ao rodar o AG. Verifique as dependencias com setup.bat.
    pause & exit /b 1
)

echo.
echo ag_data.js gerado. Abra trelica_3d.html no browser para ver a animacao.
pause
