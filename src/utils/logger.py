"""
Logger configuration для проекта.
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Настроить логирование для проекта.
    
    Args:
        name: имя логгера
    
    Returns:
        Настроенный Logger объект
    """
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Формат логов
    log_format = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler для console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    # Handler для файла (опционально)
    try:
        file_handler = logging.FileHandler(
            f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except:
        pass  # Если логирование в файл не работает, продолжаем
    
    logger.addHandler(console_handler)
    
    return logger
