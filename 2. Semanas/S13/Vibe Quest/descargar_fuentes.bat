@echo off
setlocal
cd /d "%~dp0"
set "PY="
for %%P in ("D:\Jorge Chamorro\Programas\PsychoPy\python.exe" "%LocalAppData%\Programs\Python\Python313\python.exe" "%LocalAppData%\Programs\Python\Python312\python.exe" "%LocalAppData%\Programs\Python\Python311\python.exe" "%LocalAppData%\Programs\Python\Python310\python.exe") do (
  if not defined PY if exist %%P set "PY=%%~P"
)
if not defined PY (
  echo No encontre Python. Instala Python desde python.org y marca "Add python.exe to PATH".
  pause
  exit /b 1
)
echo Usando: %PY%
"%PY%" -m pip install --upgrade pip playwright pandas
if errorlevel 1 ( echo Fallo la instalacion de paquetes. & pause & exit /b 1 )
"%PY%" -m playwright install chromium
if errorlevel 1 ( echo Fallo la instalacion de Chromium. & pause & exit /b 1 )
"%PY%" codigo\descargar_fuentes.py
echo.
echo Listo. Revisa la carpeta fuentes\ y avisame.
pause
