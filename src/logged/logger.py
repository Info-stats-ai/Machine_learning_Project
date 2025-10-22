import logging
import os
from datetime import datetime

# Global variable to track if logging is already configured
_logging_configured = False

def setup_logging():
    """Setup logging configuration"""
    global _logging_configured
    
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)
    
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
    
    # Only configure if not already configured
    if not _logging_configured:
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create a new logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        _logging_configured = True
        print(f"Logging configured. Log file: {LOG_FILE_PATH}")
    
    return logging.getLogger(__name__)

# Setup logging when module is imported
logger = setup_logging()
if __name__ == "__main__":
    logging.info("Logging has started")
