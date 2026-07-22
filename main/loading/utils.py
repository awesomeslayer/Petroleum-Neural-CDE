import torch
import numpy as np
import random
import os
import logging
from datetime import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"run_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path), # Log to the unique file
            logging.StreamHandler()             # Log to the console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Output will be saved to: {log_file_path}")
    
    return logger

def get_project_root():
    """Returns the absolute path of the project root."""
    return os.path.dirname(os.path.abspath(__file__))