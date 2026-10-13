""" Servidor web assíncrono com carregamento único do artefato no evento de lifespan e validação de rotas """

# Bibliotecas
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import BatchPredictionInput, BatchPredictionOutput, SinglePredictionOutput
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)       # obter o log do script atual

# Cache em memória do modelo
model_artifact = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Carrega o Modelo serializado ao iniciar a API e libera memória ao fechar """
    logger.info("Carregando artefato do modelo na inicialização da API")

    if not settings.MODEL_PATH.exists():
        logger.error(f"Artafato do Modelo não encontrado em {settings.MODEL_PATH}")
        raise RuntimeError("Modelo não encontrado. Treine o o Modelo antes de iniciar a API")

    model_artifact["pipeline"] = joblib.load(settings.MODEL_PATH)
    logger.info("Modelo carreado na memória com sucesso")
    yield
    model_artifact.clear()
    logger.info("Modelo removido da memória")

app = FastAPI(
    title = settings.PROJECT_NAME,
    version = settings.VERSION,
    lifespan = lifespan
)

# Rotas
@app.get("/halth", status_code=status.HTTP_200_OK)
def health_check():
    """ Endpoint de verificação de integridade da API """
    return {
        "status": "healthy",
        "model_loaded": "pipeline" in "model_artifact",
        "version": settings.VERSION
    }

@app.post("/predict", response_model=BatchPredictionOutput, status_code=status.HTTP_200_OK)
def predict_batch(payload: BatchPredictionInput):
    """ Endpoint de predição em lote para avaliação de imóveis """
    if "pipeline" not in model_artifact:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "Serviço indisponível: Modelo não inicializado"
        )

    try:
        # Converter o payload Pydantic para DataFrame do Pandas
        input_data = [item.model_dump() for item in payload.inputs]     # transformando Objetos Pydanatic em Dicionários Python
        df_input = pd.DataFrame(input_data)

        # Inferência via Pipeline
        predictions = model_artifact["pipeline"].predict(df_input)

        results = [SinglePredictionOutput(preco_estimado=round(float(pred), 2)) for pred in predictions ]

        return BatchPredictionOutput(predictions=results)

    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"Erro inteno ao processar a inferência: {str(e)}"
        )