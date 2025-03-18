import subprocess
import os
import sys
import tomli  # Or import tomllib if Python < 3.11


def start_in_conda_env(config):
    conda_env_name = config["conda_env_name"]
    python_script_path = config["python_script_path"]
    conda_base_path = config.get("conda_base_path")  # Use get method to allow conda_base_path as empty

    # Choose how to find conda path based on operating system
    if sys.platform == "win32":  # Windows
        if conda_base_path:
            activate_cmd = os.path.join(conda_base_path, "Scripts", "activate.bat")
        else:
            # Try to find conda from PATH
            for path in os.environ["PATH"].split(os.pathsep):
                possible_path = os.path.join(path, "activate.bat")
                if os.path.exists(possible_path):
                    activate_cmd = possible_path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find activate.bat. Please specify conda_base_path in config.toml."
                )
        # Use cmd /K to keep command prompt open
        command = (
            f'cmd /K ""{activate_cmd}" {conda_env_name} && python "{python_script_path}""'
        )

    elif sys.platform in ("linux", "linux2", "darwin"):  # Linux or macOS
        if conda_base_path:
            conda_bin = os.path.join(conda_base_path, "bin")
        else:
            # Try to find conda from PATH
            for path in os.environ["PATH"].split(os.pathsep):
                if os.path.exists(os.path.join(path, "conda")):
                    conda_bin = path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find conda. Please specify conda_base_path in config.toml."
                )
        # Starts with conda run
        command = f'conda run -n {conda_env_name} python "{python_script_path}"'

    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")

    # Start command prompt
    try:
        if sys.platform == "win32":
            subprocess.Popen(command, shell=True)
        elif sys.platform in ("linux", "linux2", "darwin"):
            if sys.platform == "darwin":
                terminal = "osascript"
                script = f'tell application "Terminal" to do script "{command}"'
            else:
                # Try some mainstream terminal
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
                        break
                    except FileNotFoundError:
                        pass
                else:
                    raise FileNotFoundError(
                        "Could not find a suitable terminal. Please install gnome-terminal, konsole, xterm, terminator, or xfce4-terminal."
                    )

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    # Read config.toml
    config_file_path = "config.toml"  # Set path of the config.toml
    try:
        with open(config_file_path, "rb") as f:
            config = tomli.load(f)  # Or tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file_path}")
        sys.exit(1)
    except tomli.TOMLDecodeError as e:  # Or tomllib.TOMLDecodeError
        print(f"Error decoding TOML file: {e}")
        sys.exit(1)

    # Check wheather the config exist or not
    required_keys = ["conda_env_name", "python_script_path"]
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key '{key}' in config.toml")
            sys.exit(1)

    # Call the function
    if start_in_conda_env(config):
        print(
            f"Successfully started '{config['python_script_path']}' in Conda environment '{config['conda_env_name']}'."
        )
    else:
        print("Failed to start the script.")