""" Separação do dataset, ajuste do modelo, cálculo das métricas e serialização do artefato .joblib """

# Bibliotecas
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import settings
from src.dataset import ComplexDatasetGenerator
from src.logger import get_logger
from src.model_pipeline import PipelineLinearRegressionBuilder

logger = get_logger(__name__)       # obtendo os Logs do Scipt executado

def train_and_evaluate():
    logger.info("Iniciando ciclo de treinamento ...")

    # 1. Carga de dados
    generator = ComplexDatasetGenerator()
    df = generator.generate()

    # Features / Target
    X = df.drop(columns=["preco"])
    y = df["preco"]

    # 2. Split treino/test
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=settings.RANDOM_STATE)

    # 3. Construção e ajuste do Pipeline
    pipeline = PipelineLinearRegressionBuilder.build_pipeline()
    
    logger.info("Ajustando modelo (fit)")
    pipeline.fit(X_train, y_train)

    # 4. Inferência e avaliação no teste
    y_pred = pipeline.predict(X_test)

    # métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    wape = np.sum(np.abs(y_test - y_pred) / np.sum(y_test))

    logger.info("--- MÉTRICAS DE DESEMPENHO EM TESTE ---")
    logger.info(f"RMSE: R$ {rmse:.2f}")
    logger.info(f"MAE: R$ {mae:.2f}")
    logger.info(f"WAPE: {wape: .4%}")

    # 5. Persistência do artefato
    joblib.dump(pipeline, settings.MODEL_PATH)
    logger.info(f"Modelo serializado e salvo com sucesso em: {settings.MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()