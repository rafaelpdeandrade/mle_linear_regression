""" Amostra de 100.000 registros contendo atributos contínuos, categóricos, dados faltantes e ruídos"""

import numpy as np
import pandas as pd
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

class ComplexDatasetGenerator:
    """Gera um dataset sintético de grande escala para regressão linear com desafios reais."""

    def __init__(self, n_samples: int = settings.N_SAMPLES, random_state: int = settings.RANDOM_STATE):
        self.n_samples = n_samples
        self.random_state = random_state

    def generate(self) -> pd.DataFrame:
        logger.info(f"Gerando dataset sintético complexo com {self.n_samples} registros...")
        np.random.seed(self.random_state)

        # Features numéricas
        area_m2 = np.random.normal(loc=120, scale=45, size=self.n_samples).clip(30, 600)
        idade_imovel = np.random.uniform(low=0, high=50, size=self.n_samples)
        distancia_centro_km = np.random.exponential(scale=8, size=self.n_samples).clip(0.5, 40)
        num_quartos = np.random.choice([1, 2, 3, 4, 5], size=self.n_samples, p=[0.15, 0.35, 0.35, 0.10, 0.05])

        # Features Categóricas
        tipo_imovel = np.random.choice(["Apartamento", "Casa", "Cobertura"], size=self.n_samples, p=[0.6, 0.3, 0.1])
        padrao_acabamento = np.random.choice(["Popular", "Medio", "Alto"], size=self.n_samples, p=[0.4, 0.4, 0.2])

        # Equação do Target Real (com pesos lineares e interações)
        bias = 150_000.0
        coef_area = 3_500.0
        coef_idade = -1_200.0
        coef_distancia = -2_800.0
        coef_quartos = 15_000.0

        map_tipo = {"Apartamento": 0.0, "Casa": 25_000.0, "Cobertura": 120_000.0}
        map_padrao = {"Popular": -30_000.0, "Medio": 20_000.0, "Alto": 90_000.0}

        efeito_tipo = np.vectorize(map_tipo.get)(tipo_imovel)
        efeito_padrao = np.vectorize(map_padrao.get)(padrao_acabamento)

        # Ruído Gaussiano
        ruido = np.random.normal(loc=0, scale=25_000, size=self.n_samples)

        preco_real = (
            bias
            + (area_m2 * coef_area)
            + (idade_imovel * coef_idade)
            + (distancia_centro_km * coef_distancia)
            + (num_quartos * coef_quartos)
            + efeito_tipo
            + efeito_padrao
            + ruido
        )

        df = pd.DataFrame({
            "area_m2": area_m2,
            "idade_imovel": idade_imovel,
            "distancia_centro_km": distancia_centro_km,
            "num_quartos": num_quartos,
            "tipo_imovel": tipo_imovel,
            "padrao_acabamento": padrao_acabamento,
            "preco": preco_real
        })

        # Injetando valores nulos para simular sujeira de produção (5% de nulos em algumas colunas)
        mask_null_idade = np.random.rand(self.n_samples) < 0.05
        df.loc[mask_null_idade, "idade_imovel"] = np.nan

        mask_null_padrao = np.random.rand(self.n_samples) < 0.03
        df.loc[mask_null_padrao, "padrao_acabamento"] = np.nan

        logger.info("Dataset sintético gerado com sucesso.")

        # Exportando o Dataset
        return df.to_csv(settings.DATA_DIR / "df_sintetico.csv", index=False, encoding="UTF-8")

if __name__ == "__main__":
    generator = ComplexDatasetGenerator()
    generator.generate()