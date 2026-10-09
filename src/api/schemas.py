""" Define as regras de validação rigorosa para os dados de entrada e saída HTTP"""

# Bibliotecas
from typing import List, Optional
from pydantic import BaseModel, Field

class PredictionImput(BaseModel):
    area_m2: float = Field(..., gt=10, lt=2000, description="Área útil do imóvel em metros quadrados")
    idade_imovel: Optional[float] = Field(default=None, ge=0, le=120, description="Idade da construção em anos")
    distancia_centro_km: float = Field(..., ge=0, le=200, description="Distância até o centro urbano em km")
    num_quartos: int = Field(..., ge=1, le=10, description="Quantidade de quartos")
    tipo_imovel: str = Field(..., example="Apartamento", description="Tipo de construção (Apartamento, Casa, Cobertura)")
    padrao_acabamento: Optional[str] = Field(default="Medio", example="Medio", descrition="Padrão de acabamento (Popular, Medio, Alto)")

class BatchPredictionInput(BaseModel):
    inputs: List[PredictionImput]

class SinglePredictionOutput(BaseModel):
    preco_estimado: float = Field(..., description="Valor predito do imóvel em Reais (BRL)")

class BatchPredictionOutput(BaseModel):
    predictions: List[SinglePredictionOutput]