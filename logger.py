# logger.py
import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 全局控制台 handler，只创建一次
_console_handler = None
def get_console_handler(level=logging.DEBUG):
    global _console_handler
    if _console_handler is None:
        _console_handler = logging.StreamHandler(sys.stdout)
        _console_handler.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _console_handler.setFormatter(formatter)
    return _console_handler


def _get_daily_file_handler(filename: str, level=logging.DEBUG):
    """
    返回一个按日期切分的 handler。
    文件格式：
        当前文件：filename.log
        归档文件：filename.2025-11-15.log
    """
    log_path = LOG_DIR / filename

    handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=False
    )

    # ⭐ 关键：设置归档文件名格式
    handler.suffix = "%Y-%m-%d.log"

    handler.setLevel(level)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    return handler


def get_logger(name: str, filename: str = None, level=logging.DEBUG) -> logging.Logger:
    """
    返回一个 logger
    name: 模块名，一般用 __name__
    filename: 日志文件名，不含路径。如 "agent.log"
    level: 日志等级
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        
        # 1. 控制台
        logger.addHandler(get_console_handler(level))

        # 2. 文件按日期拆分
        if filename:
            logger.addHandler(_get_daily_file_handler(filename, level))

    return logger
