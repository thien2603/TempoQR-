"""
Configuration settings for TempoQR API
"""
 
from typing import Optional
import os
 
class Settings:
    """Main configuration class for TempoQR API"""
 
    # App settings
    APP_NAME: str = "TempoQR API"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
 
    # Model settings
    MODEL_PATH: str = "models/models/wikidata_big/qa_models/tempoqr_full_export.pt"
    TKBC_PATH: str = "models/models/wikidata_big/kg_embeddings/tcomplex.ckpt"
    DEVICE: str = "cpu"
    DATASET_NAME: str = "wikidata_big"
    TOP_K: int = 5
 
    # Data paths
    DATA_ROOT: str = "data/data"
    KG_FILE: str = "full.txt"
    DICT_FOLDER: str = "tkbc_processed_data"
 
    # API settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
 
    # Model parameters
    MODEL_TYPE: str = "tempoqr"
    SUPERVISION: str = "soft"
    FUSE: str = "add"
    EXTRA_ENTITIES: bool = False
    FROZEN: int = 1
    LM_FROZEN: int = 1
    CORRUPT_HARD: float = 0.0
 
    # Tokenizer settings
    TOKENIZER_NAME: str = "distilbert-base-uncased"
    MAX_LENGTH: int = 512
 
    # Performance settings
    BATCH_SIZE: int = 32
    MAX_ENTITIES: int = 10
    MAX_TIMES: int = 10
 
    # Project root
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def __init__(self):
        # Load from environment variables if needed
        self.APP_NAME = os.getenv("APP_NAME", self.APP_NAME)
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.VERSION = os.getenv("VERSION", self.VERSION)
        self.MODEL_PATH = os.getenv("MODEL_PATH", self.MODEL_PATH)
        self.TKBC_PATH = os.getenv("TKBC_PATH", self.TKBC_PATH)
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        self.DATASET_NAME = os.getenv("DATASET_NAME", self.DATASET_NAME)
        self.TOP_K = int(os.getenv("TOP_K", str(self.TOP_K)))
        self.HOST = os.getenv("HOST", self.HOST)
        self.PORT = int(os.getenv("PORT", str(self.PORT)))

# Global settings instance
settings = Settings()
 
def get_full_path(path: str) -> str:
    """Get full path relative to project root"""
    return os.path.join(settings.PROJECT_ROOT, path)
 
# Model configuration
class ModelConfig:
    """Model-specific configuration"""
 
    def __init__(self):
        self.model_path = get_full_path(settings.MODEL_PATH)
        self.tkbc_path = get_full_path(settings.TKBC_PATH)
        self.device = settings.DEVICE
        self.dataset_name = settings.DATASET_NAME
        self.top_k = settings.TOP_K
 
        # Model arguments
        self.args = {
            'model': settings.MODEL_TYPE,
            'supervision': settings.SUPERVISION,
            'fuse': settings.FUSE,
            'extra_entities': settings.EXTRA_ENTITIES,
            'frozen': settings.FROZEN,
            'lm_frozen': settings.LM_FROZEN,
            'corrupt_hard': settings.CORRUPT_HARD,
            'dataset_name': settings.DATASET_NAME,
            'tkg_file': get_full_path(f"{settings.DATA_ROOT}/{settings.DATASET_NAME}/kg/{settings.KG_FILE}"),
            'batch_size': settings.BATCH_SIZE,
            'lr': 2e-4
        }
 
# Global model config
model_config = ModelConfig()