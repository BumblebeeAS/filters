import logging


class RclLogHandler(logging.Handler):
    """logging Handler class to remove the need for dependency on rclpy."""

    def __init__(self, logger, name):
        super().__init__()
        self.logger = logger
        self._name = name
        self.formatter = logging.Formatter(
            fmt="%(message)s\t %(filename)s:%(lineno)d",
            datefmt="%d/%m %H:%M:%S",
        )

    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.FATAL:
            self.logger.fatal(msg)
        elif record.levelno >= logging.ERROR:
            self.logger.error(msg)
        elif record.levelno >= logging.WARNING:
            self.logger.warn(msg)
        elif record.levelno >= logging.INFO:
            self.logger.info(msg)
        elif record.levelno >= logging.DEBUG:
            self.logger.debug(msg)
        else:
            self.logger.trace(msg)
