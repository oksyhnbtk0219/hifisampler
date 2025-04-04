import logging
import yaml
import shutil
from pathlib import Path

def load_config_from_yaml(script_path: Path):
    def decorator(cls):
        try:
            script_dir = script_path.parent
            config_path = script_dir / 'config.yaml'
            default_config_path = script_dir / 'config.default.yaml'

            if not config_path.exists():
                if default_config_path.exists():
                    logging.info(f"Copying {default_config_path} to {config_path}")
                    shutil.copy(default_config_path, config_path)
                else:
                    logging.warning(f"Default config {default_config_path} not found.")

            if config_path.exists():
                logging.info(f"Loading config from: {config_path}")
                with open(config_path, 'r', encoding='utf-8') as f:
                    nested_config = yaml.safe_load(f) or {}

                logging.info(f"Loaded nested config: {nested_config}")

                for section_name, section_data in nested_config.items():
                    if isinstance(section_data, dict):
                        for key, value in section_data.items():
                            if hasattr(cls, key):
                                setattr(cls, key, value) 
                            else:
                                logging.warning(f"Skipping key '{key}' from YAML section '{section_name}' as it's not defined in class {cls.__name__}.")
            else:
                logging.warning(f"Config file {config_path} not found. Using hardcoded defaults for {cls.__name__}.")

        except Exception as e:
            logging.error(f"Error applying config loader decorator to {cls.__name__}: {e}", exc_info=True)

        logging.info(f"--- Finished applying decorator to {cls.__name__} ---")
        return cls
    return decorator
