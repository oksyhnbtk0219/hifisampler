from pathlib import Path
import shutil
import subprocess
import os
import sys
import yaml


def get_default_conda_base_path():
    """Attempts to find the default conda base path."""
    if sys.platform == "win32":
        # Common installation locations on Windows
        possible_paths = [
            os.path.expanduser(r"~\miniconda3"),
            os.path.expanduser(r"~\anaconda3"),
            r"C:\miniconda3",
            r"C:\anaconda3",
            r"C:\ProgramData\Miniconda3",
            r"C:\ProgramData\Anaconda3"
        ]
    elif sys.platform in ("linux", "linux2", "darwin"):
        # Common installation locations on Linux/macOS
        possible_paths = [
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            "/opt/miniconda3",
            "/opt/anaconda3",
        ]
    else:
        return None

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def start_in_conda_env(config):
    conda_env_name = config["conda_env_name"]
    python_script_path = config["python_script_path"]
    conda_base_path = config.get("conda_base_path")  # Use get method to allow conda_base_path as empty

    # Choose how to find conda path based on operating system
    if sys.platform == "win32":  # Windows
        if conda_base_path:
            activate_cmd = os.path.join(conda_base_path, "Scripts", "activate.bat")
            if not os.path.exists(activate_cmd):
                print(f"Warning: activate.bat not found at: {activate_cmd}. Trying to find default conda path.")
                conda_base_path = get_default_conda_base_path()
                if conda_base_path:
                    activate_cmd = os.path.join(conda_base_path, "Scripts", "activate.bat")
                    if not os.path.exists(activate_cmd):
                        raise FileNotFoundError(f"activate.bat not found at: {activate_cmd}. Please check conda installation.")
                else:
                    raise FileNotFoundError("Could not find default conda installation. Please specify conda_base_path in config.yaml or ensure conda is in your PATH.")
        else:
            # Try to find conda from PATH
            activate_cmd = None
            for path in os.environ["PATH"].split(os.pathsep):
                possible_path = os.path.join(path, "activate.bat")
                if os.path.exists(possible_path):
                    activate_cmd = possible_path
                    break
            if activate_cmd is None:
                print("Warning: Could not find activate.bat in PATH. Trying to find default conda path.")
                conda_base_path = get_default_conda_base_path()
                if conda_base_path:
                    activate_cmd = os.path.join(conda_base_path, "Scripts", "activate.bat")
                    if not os.path.exists(activate_cmd):
                        raise FileNotFoundError(f"activate.bat not found at: {activate_cmd}. Please check conda installation.")
                else:
                    raise FileNotFoundError(
                        "Could not find activate.bat in PATH or default conda installation. Please specify conda_base_path in config.yaml or ensure conda is in your PATH."
                    )
        # Use cmd /K to keep command prompt open
        command = (
            f'cmd /K ""{activate_cmd}" {conda_env_name} && python "{python_script_path}""'
        )

    elif sys.platform in ("linux", "linux2", "darwin"):  # Linux or macOS
        if conda_base_path:
            conda_bin = os.path.join(conda_base_path, "bin", "conda")
            if not os.path.exists(conda_bin):
                print(f"Warning: conda not found at: {conda_bin}. Trying to find default conda path.")
                conda_base_path = get_default_conda_base_path()
                if conda_base_path:
                    conda_bin = os.path.join(conda_base_path, "bin", "conda")
                    if not os.path.exists(conda_bin):
                        raise FileNotFoundError(f"conda not found at: {conda_bin}. Please check conda installation.")
                else:
                    raise FileNotFoundError("Could not find default conda installation. Please specify conda_base_path in config.yaml or ensure conda is in your PATH.")
        else:
            # Try to find conda from PATH
            conda_bin = None
            for path in os.environ["PATH"].split(os.pathsep):
                possible_path = os.path.join(path, "conda")
                if os.path.exists(possible_path):
                    conda_bin = possible_path
                    break
            if conda_bin is None:
                print("Warning: Could not find conda in PATH. Trying to find default conda path.")
                conda_base_path = get_default_conda_base_path()
                if conda_base_path:
                    conda_bin = os.path.join(conda_base_path, "bin", "conda")
                    if not os.path.exists(conda_bin):
                        raise FileNotFoundError(f"conda not found at: {conda_bin}. Please check conda installation.")
                else:
                    raise FileNotFoundError(
                        "Could not find conda in PATH or default conda installation. Please specify conda_base_path in config.yaml or ensure conda is in your PATH."
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
                subprocess.Popen([terminal, "-e", script])
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
    # Read config.yaml
    config_file_path = Path("./config.yaml")  # Set path of the config.yaml
    default_config_file_path = Path("./config.default.yaml")
    if not config_file_path.exists():
        if default_config_file_path.exists():
            try:
                shutil.copy(default_config_file_path, config_file_path)
            except Exception as e:
                print(f"Error copying default config file: {e}")
                sys.exit(1)
        else:
            print(f"Error: Default config file not found: {default_config_file_path}")
            sys.exit(1)

    env_config = {}

    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            loaded_config = yaml.safe_load(f)  # Load yaml file

            if loaded_config is None:
                 print(f"Warning: Configuration file '{config_file_path}' is empty or invalid. Cannot proceed without environment configuration.")
                 sys.exit(1)
            elif isinstance(loaded_config, dict):
                env_config_value = loaded_config.get("env")
                if env_config_value is None:
                     print(f"Error: 'env' key not found or its value is null in '{config_file_path}'. Cannot proceed without environment configuration.")
                     sys.exit(1)
                elif isinstance(env_config_value, dict):
                    env_config = env_config_value
                else:
                    print(f"Error: The value under 'env' key in '{config_file_path}' is not a dictionary.")
                    sys.exit(1)
            else:
                print(f"Error: Configuration file '{config_file_path}' does not contain a dictionary at the top level.")
                sys.exit(1)

    except FileNotFoundError:
        # This case is less likely due to the initial check, but kept for robustness
        print(f"Error: Config file not found: {config_file_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        # Error decoding yaml file
        print(f"Error decoding yaml file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while processing '{config_file_path}': {e}")
        sys.exit(1)


    # Check whether the config exist or not
    required_keys = ["conda_env_name", "python_script_path"]
    missing_keys = [key for key in required_keys if key not in env_config]
    if missing_keys:
        print(f"Error: Missing required key(s) in the 'env' section of config.yaml: {', '.join(missing_keys)}")
        sys.exit(1)

    # Check if python script exists
    script_path_str = str(env_config["python_script_path"])
    if not os.path.exists(script_path_str):
        print(f"Error: Python script not found: {script_path_str}")
        sys.exit(1)

    # Call the function
    if start_in_conda_env(env_config):
        print(
            f"Successfully attempted to start '{script_path_str}' in Conda environment '{env_config['conda_env_name']}'."
        )
    else:
        print("Failed to start the script.")
