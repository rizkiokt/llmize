import logging
import time
from colorama import Fore, Style, init

# Initialize colorama for colored console output
init(autoreset=True, strip=False)

# Define color mapping for different log levels
LOG_COLORS = {
    "DEBUG": Fore.CYAN,    # Light Blue
    "WARNING": Fore.YELLOW,  # Yellow
    "ERROR": Fore.RED,     # Red
    "CRITICAL": Fore.MAGENTA + Style.BRIGHT  # Bright Magenta
}
class ColoredFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Use the custom datefmt if provided
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
            return f"{s}.{int(record.msecs):03d}"
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            return f"{t}.{int(record.msecs):03d}"

    def format_with_time(self, record):
        if record.levelname == "INFO":
            fmt = "%(asctime)s - %(message)s"
        else:
            fmt = "%(asctime)s - %(levelname)s - %(message)s"
        temp_formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        log_color = LOG_COLORS.get(record.levelname, Fore.WHITE)
        formatted_message = temp_formatter.format(record)
        return f"{log_color}{formatted_message}{Style.RESET_ALL}"
    
    def format(self, record):
        if record.levelname == "INFO":
            fmt = "%(message)s"
        else:
            fmt = "%(levelname)s - %(message)s"
        temp_formatter = logging.Formatter(fmt)
        log_color = LOG_COLORS.get(record.levelname, Fore.WHITE)
        formatted_message = temp_formatter.format(record)
        return f"{log_color}{formatted_message}{Style.RESET_ALL}"



# Initialize the logger
logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)  # Default to INFO, can be changed dynamically

# Remove all existing handlers (prevents duplicate logs)
logger.handlers.clear()
logger.propagate = False


# Console handler with color support
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))  # Removed filename

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
