# Bibliotecas
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)       # obtendo os logs do arquivo executado

class PipelineLinearRegressionBuilder():
    """ Constrói o Pipeline do Scikit-Learn isolando o pré-processamento e o estimador. """

    @staticmethod   # função estática: não depende do estado da instância (self) e da Classe (cls)
    def build_pipeline() -> Pipeline:
        logger.info("Construindo Pipeline de Pré-Processamento e o Modelo...")

        # Features
        num_features = ["area_m2", "idade_imovel", "distancia_centro_km", "num_quartos"]
        cat_features = ["tipo_imovel", "padrao_acabamento"]

        # Pipeline para atributos numéricos (Imputação pela Mediana + Padronização)
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler())
        ])

        # Pipeline para atributos categóricos (Imputação de categoria mais frequente + One Hot)
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", spase_output=False))
        ])

        # Combinador de colunas
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ])

        # Pipeline final unificando 'Pré-processamento' e 'Estimador Regularizado' (Ridge)
        full_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=settings.ALPHA, random_state=settings.RANDOM_STATE))
        ])

        return full_pipeline

