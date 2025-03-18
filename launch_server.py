import subprocess
import os
import sys
import tomli  # 或 import tomllib 如果Python < 3.11


def start_in_conda_env(config):
    """
    在指定的 Conda 环境中启动一个新的持久化 CMD 窗口，并在其中运行 Python 脚本。

    Args:
        config: 包含配置信息的字典。
    """

    conda_env_name = config["conda_env_name"]
    python_script_path = config["python_script_path"]
    conda_base_path = config.get("conda_base_path")  # 使用 get 方法，允许 conda_base_path 为空

    # 1. 确定操作系统和 Conda 激活方式
    if sys.platform == "win32":  # Windows
        if conda_base_path:
            activate_cmd = os.path.join(conda_base_path, "Scripts", "activate.bat")
        else:
            # 尝试从 PATH 中查找 conda
            for path in os.environ["PATH"].split(os.pathsep):
                possible_path = os.path.join(path, "activate.bat")
                if os.path.exists(possible_path):
                    activate_cmd = possible_path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find activate.bat. Please specify conda_base_path in config.toml."
                )
        # 使用 cmd /K 来保持窗口打开
        command = (
            f'cmd /K ""{activate_cmd}" {conda_env_name} && python "{python_script_path}""'
        )

    elif sys.platform in ("linux", "linux2", "darwin"):  # Linux or macOS
        if conda_base_path:
            conda_bin = os.path.join(conda_base_path, "bin")
        else:
            # 尝试从PATH中寻找conda
            for path in os.environ["PATH"].split(os.pathsep):
                if os.path.exists(os.path.join(path, "conda")):
                    conda_bin = path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find conda. Please specify conda_base_path in config.toml."
                )
        # 使用conda run
        command = f'conda run -n {conda_env_name} python "{python_script_path}"'

    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")

    # 2. 启动新的 CMD 窗口或终端
    try:
        if sys.platform == "win32":
            subprocess.Popen(command, shell=True)
        elif sys.platform in ("linux", "linux2", "darwin"):
            if sys.platform == "darwin":
                terminal = "osascript"
                script = f'tell application "Terminal" to do script "{command}"'
            else:  # linux
                # 尝试几种常见的终端
                terminals = [
                    "gnome-terminal",
                    "konsole",
                    "xterm",
                    "terminator",
                    "xfce4-terminal",
                ]
                for terminal in terminals:
                    try:
                        subprocess.Popen([terminal, "-e", command])
                        break  # 如果成功打开就跳出循环
                    except FileNotFoundError:
                        pass  # 如果没找到，则继续尝试下一个
                else:
                    raise FileNotFoundError(
                        "Could not find a suitable terminal. Please install gnome-terminal, konsole, xterm, terminator, or xfce4-terminal."
                    )

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    # 读取 TOML 配置文件
    config_file_path = "config.toml"  # 配置文件路径
    try:
        with open(config_file_path, "rb") as f:  # 以二进制模式打开
            config = tomli.load(f)  # 或 tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file_path}")
        sys.exit(1)
    except tomli.TOMLDecodeError as e:  # 或 tomllib.TOMLDecodeError
        print(f"Error decoding TOML file: {e}")
        sys.exit(1)

    # 检查配置项是否存在
    required_keys = ["conda_env_name", "python_script_path"]
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key '{key}' in config.toml")
            sys.exit(1)

    # 调用函数
    if start_in_conda_env(config):
        print(
            f"Successfully started '{config['python_script_path']}' in Conda environment '{config['conda_env_name']}'."
        )
    else:
        print("Failed to start the script.")