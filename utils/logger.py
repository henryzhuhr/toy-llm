import logging
from logging import FileHandler, Filter, Formatter, Handler, StreamHandler, getLogger
from logging.handlers import RotatingFileHandler
from typing import List, Optional

DEFAULT_FORMAT_STR = (
    "[%(asctime)s]" " | %(levelname)s" " | %(pathname)s:%(lineno)d" " | %(message)s"
)


class ColoredFormatter(Formatter):
    """自定义彩色日志格式化器"""

    COLORS = {
        'DEBUG': '\033[01;36m',  # 青色
        'INFO': '\033[01;32m',  # 绿色
        'WARNING': '\033[01;33m',  # 黄色
        'ERROR': '\033[01;31m',  # 红色
        'CRITICAL': '\033[01;95m',  # 粉红色（亮 magenta）
        'RESET': '\033[01;0m',  # 重置颜色
    }

    def format(self, record: logging.LogRecord):
        original_format = super().format(record)
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        return f"{color}{original_format}{reset_color}"


class CLogger:
    """自定义日志记录类，支持控制台和文件双输出

    Attributes:
        name (str): 日志记录器名称
        level (int): 全局日志级别 (默认INFO)
        console_level (int): 控制台输出级别 (默认与全局一致)
        file_level (int): 文件输出级别 (默认与全局一致)
        log_file (str): 日志文件路径 (默认不启用文件日志)
        max_bytes (int): 单个日志文件最大字节数 (默认5MB)
        backup_count (int): 最多保留的日志文件数 (默认3)
        format_str (str): 日志格式字符串
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        console_level: int = logging.DEBUG,
        file_level: Optional[int] = logging.INFO,
        log_file: Optional[str] = None,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
    ):
        self.logger = getLogger(name)
        self.logger.setLevel(level)
        self.handlers: List[logging.Handler] = []  # 用于存储处理器

        if console_level:
            self.add_stream_handler(console_level)
        if log_file:
            self.add_file_handler(log_file, file_level, max_bytes, backup_count)

    def get_logger(self):
        """获取配置好的日志记录器"""
        return self.logger

    def set_level(self, level: int = logging.INFO):
        """动态设置日志级别"""
        self.logger.setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)

    def add_stream_handler(
        self, level: int = logging.DEBUG, format_str=DEFAULT_FORMAT_STR
    ):
        """初始化控制台处理器"""
        handler = StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(ColoredFormatter(format_str))
        self.__add_handler(handler)

    def add_file_handler(
        self,
        log_file: str,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        format_str=DEFAULT_FORMAT_STR,
    ):
        """初始化文件处理器"""
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        handler.setLevel(level)
        handler.setFormatter(Formatter(format_str))
        self.__add_handler(handler)

    def __add_handler(self, handler: Handler):
        """添加唯一处理器防止重复"""
        handler_type = type(handler)
        if any(isinstance(h, handler_type) for h in self.handlers):
            self.logger.warning(f"已经添加了类型为 {handler_type} 的处理器")
            return  # 如果已经添加了相同类型的处理器，则直接返回

        self.handlers.append(handler)
        self.logger.addHandler(handler)


# 创建日志记录器
clogger = CLogger(__name__)
logger = clogger.get_logger()

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 记录日志
    logger.info("这是一条info日志")
    logger.debug("这是一条debug日志")  # 只会出现在文件中
    logger.warning("这是一条warning日志")
    logger.error("这是一条error日志")
    logger.critical("这是一条critical日志")
