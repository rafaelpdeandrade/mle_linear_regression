# Bibliotecas
import logging
import sys
from src.config import settings

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Se não existir o Logger, iremos criar 
    if not logger.handlers:
        logger.setLevel(logging.INFO)       # inserir apenas informações
        formatter = logging.Formatter(      # definindo as informações que serão visualizadas
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        )

        # Handler Console
        console_handler = logging.StreamHandler(sys.stdout) # saída enviada para o Terminal
        console_handler.setFormatter(formatter)             # formatando o log
        logger.addHandler(console_handler)                  # adicionando as informações do console no log

        # Handler Arquivo
        file_handler = logging.FileHandler(settings.LOGS_DIR / "mle_pipeline.log")  # local para armazenar o arquivo de log
        file_handler.setFormatter(formatter)                                        # formatadndo o log
        
        logger.addHandler(file_handler)                                             # adicionando o aqruivo no Logg

    return logger

