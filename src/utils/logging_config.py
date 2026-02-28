"""
Centralized logging configuration with JSON structured logging
"""
import logging
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger
import os


def setup_logging(log_level: str = None):
    """
    Setup centralized logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    
    # JSON formatter
    json_formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s %(lineno)d',
        rename_fields={'level': 'severity', 'timestamp': '@timestamp'}
    )
    
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            log_record['@timestamp'] = datetime.utcnow().isoformat()
            log_record['severity'] = record.levelname
            if record.exc_info:
                log_record['exception'] = self.formatException(record.exc_info)
    
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(severity)s %(name)s %(message)s'
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log startup
    logger.info("Logging configured", extra={
        'log_level': log_level,
        'format': 'JSON'
    })
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
