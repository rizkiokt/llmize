import logging
from colorama import Fore, Style, init

# Initialize colorama for colored console output
init(autoreset=True)

# Define color mapping for different log levels
LOG_COLORS = {
    "DEBUG": Fore.CYAN,    # Light Blue
    "INFO": Fore.GREEN,    # Green
    "WARNING": Fore.YELLOW,  # Yellow
    "ERROR": Fore.RED,     # Red
    "CRITICAL": Fore.MAGENTA + Style.BRIGHT  # Bright Magenta
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter to apply colors based on log level."""
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, Fore.WHITE)
        log_message = super().format(record)
        return f"{log_color}{log_message}{Style.RESET_ALL}"

# Initialize the logger
logger = logging.getLogger("Logger")
logger.setLevel(logging.INFO)  # Default to INFO, can be changed dynamically

# Console handler with color support
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s"))

# Remove existing handlers if any (prevents duplicate logs)
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(console_handler)

# Logging functions for easy access
def log_debug(message):
    logger.debug(message)

def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

def log_critical(message):
    logger.critical(message)

# Function to dynamically set the logging level
def set_log_level(level):
    """
    Set the logging level dynamically.

    :param level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    logger.setLevel(level)
