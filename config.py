"""
Configurations for RAG_SPM project
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # General Configuration
    PROJECT_NAME = "RAG_SPM"
    VERSION = "1.0"

    # Data Paths
    DATA_DIR = os.path.join(os.getcwd(), "data")
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train_data.xlsx")
    TEST_DATA_FILE = os.path.join(DATA_DIR, "test_data.xlsx")

    # Model Parameters
    VECTOR_SIZE = 100
    WINDOW_SIZE = 5
    MIN_COUNT = 1
    SG_MODE = 1  # 1 for skip-gram, 0 for cbow

    # Knowledge Base Settings
    KB_PASSAGE_LENGTH = 500
    KB_CONTEXT_LENGTH = 200

    # Training Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01

    # Evaluation Metrics
    METRICS = ["accuracy", "f1_score"]

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = os.path.join(os.getcwd(), f"{PROJECT_NAME}.log")

    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Word2Vec Model File
    WORD2VEC_MODEL_FILE = os.path.join(DATA_DIR, f"word2vec_{VECTOR_SIZE}d.model")

    # Faiss GPU Index Path
    FAISS_INDEX_PATH = os.path.join(DATA_DIR, f"faiss_index_{VECTOR_SIZE}d.faiss")

    @classmethod
    def get(cls, key):
        return getattr(cls, key)

    @classmethod
    def set(cls, key, value):
        setattr(cls, key, value)

# Load environment-specific configurations
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
if ENVIRONMENT == "production":
    from .config_production import ProductionConfig
elif ENVIRONMENT == "staging":
    from .config_staging import StagingConfig
else:
    from .config_development import DevelopmentConfig

CONFIG = globals()[f"{ENVIRONMENT.capitalize()}Config"]()
