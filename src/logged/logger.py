import logging
import os
from datetime import datetime


def setup_logging():
    """Setup logging configuration"""
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)
    
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True  # This forces reconfiguration
    )
    
    return logging.getLogger(__name__)

# Setup logging when module is imported
logger = setup_logging()
if __name__ == "__main__":
    logging.info("Logging has started")
