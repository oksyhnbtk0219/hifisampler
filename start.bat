@echo off
REM 如果存在venv目录，则执行uv venv，并将.venv目录下的所有文件复制到venv目录下，冲突则覆盖，否则不执行任何操作，最后将venv目录更名为.venv
if exist venv (
    echo "Creating venv directory..."
    uv venv
    xcopy /s /e /y .venv\* venv\
    rmdir /s /q .venv
    ren venv .venv
)

uv run launch_server.py