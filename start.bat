@echo off
chcp 65001 >nul
echo ========================================
echo   图片素材库 - 以图搜图与去重系统
echo ========================================
echo.
echo 正在启动服务...
echo.

cd /d "%~dp0backend"
..\venv\Scripts\python.exe app.py

pause
