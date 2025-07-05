@echo off
setlocal

rem --- 设置和验证 uv 环境 ---
call :SetupUv
if %errorlevel% neq 0 (
    echo.
    echo Fatal error during uv setup. The script cannot continue.
    pause
    goto :eof
)

rem --- 启动服务 ---
uv run launch_server.py

endlocal
goto :eof


:SetupUv
    echo --- Checking for uv...
    
    rem 尝试直接执行 uv，如果成功，说明一切就绪，直接返回
    uv --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo uv is already installed and accessible.
        exit /b 0
    )

    rem 如果命令失败，则开始安装流程
    echo uv not found or not working. Attempting to install...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo Error: Failed to execute uv installation script. Please check your internet connection or PowerShell settings.
        exit /b 1
    )

    rem 安装后，更新当前会话的 PATH
    echo Installation script finished. Updating PATH for this session...
    set "UV_BIN_PATH=%USERPROFILE%\.local\bin"
    set "PATH=%UV_BIN_PATH%;%PATH%"

    rem 再次验证，确保安装和路径设置都已生效
    echo Verifying installation...
    uv --version
    if %errorlevel% neq 0 (
        echo Error: uv was installed but is still not working after PATH update.
        echo Please check the installation path is correct: %UV_BIN_PATH%
        exit /b 1
    )

    echo Verification successful!
    exit /b 0