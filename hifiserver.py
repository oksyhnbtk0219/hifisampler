import logging
import sys
import tempfile
from pathlib import Path
from filelock import FileLock, Timeout
from backend import models
from backend import server

version = '0.0.6-hifisampler'

if __name__ == '__main__':
    lock_file_path = Path(tempfile.gettempdir()) / 'hifisampler.lock'

    try:
        with FileLock(str(lock_file_path), timeout=0.5):
            logging.info(f"hifisampler {version}")
            logging.info(
                f"Successfully acquired server lock: {lock_file_path}")
            models.initialize_models()
            server.run()
    except Timeout:
        logging.warning(
            f"Another instance seems to be running (lock file '{lock_file_path}' is held). Exiting.")
        sys.exit(0)
    except Exception as e:
        logging.error(
            f"Failed to initialize or start the server: {e}", exc_info=True)
        sys.exit(1)
parser.add_argument('--whisper', action='store_true', help='enable constant whisper processing')
parser.add_argument('--whisper-preserve-harmonics', type=float, default=0.01)

