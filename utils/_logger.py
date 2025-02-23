from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO


class Logger:
    def __init__(self, name: str, level: int = INFO):
        self.logger = getLogger(name)
        self.logger.setLevel(level)
        handler = StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger
