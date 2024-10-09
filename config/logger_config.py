import yaml
import logging

class LoggerConfig:
    @staticmethod
    def load_config():
        try:
            with open("./config/config.yaml", "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    @staticmethod
    def setup_logger(name):
        config = LoggerConfig.load_config()
        logging_config = config.get('logging', {})
        logging_enabled = logging_config.get('enabled', False)
        logging_level = logging_config.get('level', 'INFO')

        logger = logging.getLogger(name)

        if logging_enabled:
            logging.basicConfig(level=logging_level)
        else:
            logger.disabled = True

        return logger