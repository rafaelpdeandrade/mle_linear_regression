# Bibliotecas
from pathlib import Path
from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Configuração moderna para o Pydantic V2
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Linear Regression MLE"
    VERSION: str = "1.0.0"

    # Caminhos (utilizando ClassVar para o Pydantic ignorar como campos de ambiente)
    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent                 # Raiz: mle_linear_regression
    ARTIFACTS_DIR: ClassVar[Path] = BASE_DIR / "artifacts"
    LOGS_DIR: ClassVar[Path] = BASE_DIR / "logs"
    DATA_DIR: ClassVar[Path] = BASE_DIR / "dataset"                                   # dados 
    MODEL_PATH: ClassVar[Path] = ARTIFACTS_DIR / "linear_regression_pipeline.joblib"  # armazenando o Modelo
    
    # Parâmetros do Dataset
    N_SAMPLES: int = 100_000
    RANDOM_STATE: int = 42

    # Parâmetros de Regularização (Ridge Linear Regression)
    ALPHA: float = 1.0

settings = Settings()

# Criando os Diretórios
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)