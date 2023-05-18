import logging
import sys
from logging.handlers import RotatingFileHandler
LOG_FORMAT ="%(asctime)s — %(name)s — %(funcName)s() — %(levelname)s — %(message)s"
LOG_FILE = "ratsql_code.log"

def get_logger(logger_name):
   logger = logging.Logger(logger_name)
   logger.setLevel(logging.DEBUG)
   file_handler = RotatingFileHandler(filename=LOG_FILE, mode='a')
   file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
   logger.addHandler(file_handler)
   return logger