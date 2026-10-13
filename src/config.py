""" Centralizar os parâmetros e caminhos do projeto de forma fortemente tipada """

# Bibliotecas
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Linear Regression MLE"
    VERSION: str = "1.0.0"

    # Caminhos
    BASE_DIR: Path = Path(__file__).resolve().parent.parent                 # Raiz: mle_linear_regression
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    LOGS_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "dataset"                                   # dados 
    MODEL_PATH: Path = ARTIFACTS_DIR / "linear_regression_pipeline.joblib"  # armazenando o Modelo
    

    # Parâmetros do Dataset
    N_SAMPLES: int = 100_000
    RANDOM_STATE: int = 42

    # Parâmetros de Regularização (Ridge Linear Regression)
    ALPHA: float = 1.0

    class Config:
        env_file = ".env"

settings = Settings()

# Criando os Diretórios
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)