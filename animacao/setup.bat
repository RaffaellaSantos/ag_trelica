@echo off
cd /d "%~dp0"
echo ============================================================
echo  Setup — Animacao Trelica Howe 3D
echo ============================================================
echo.

echo [1/3] Criando ambiente virtual (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo ERRO: Python nao encontrado. Instale em https://python.org
    pause & exit /b 1
)

echo [2/3] Instalando dependencias...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\pip.exe install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERRO ao instalar dependencias.
    pause & exit /b 1
)

echo [3/3] Baixando three.min.js (r128)...
.venv\Scripts\python.exe -c ^
  "import urllib.request; urllib.request.urlretrieve('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js','three.min.js'); print('three.min.js salvo.')"
if errorlevel 1 (
    echo AVISO: Falha ao baixar three.min.js. A animacao usara CDN automaticamente.
)

echo.
echo ============================================================
echo  Pronto! Proximos passos:
echo    run_export.bat  — gera ag_data.js com dados do AG
echo    run_viewer.bat  — abre trelica_3d.html no browser
echo ============================================================
pause
