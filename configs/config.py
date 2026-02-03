
import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

    # Embedding: Fast and good for code
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Re-ranker: High precision for final selection
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM: Using Ollama 
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "phi"  # You can switch to "llama3", "codellama", "deepseek-coder" etc.
    LLM_TEMPERATURE = 0.1
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 100
    
    EXCLUDE_PATTERNS = [
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/.venv/**",
        "**/venv/**",           # Virtual environment
        "**/env/**",            # Alternative venv name
        "**/*.pyc",
        "**/*.pyo",
        "**/.DS_Store",
        "**/*.lock",
        "**/*.log",
        "**/chroma_db/**",      
        "**/.ipynb_checkpoints/**",  
        "**/*.ipynb",          
        "**/site-packages/**",  
        "**/.pytest_cache/**",
        "**/.mypy_cache/**",
        "**/dist/**",
        "**/build/**",
        "**/.tox/**"
    ]
    
    # -------------------------------------------------------------------------
    # RETRIEVAL TUNING
    # -------------------------------------------------------------------------
    INITIAL_K = 50                 # How many docs to retrieve from Vector DB
    VECTOR_DISTANCE_THRESHOLD = 1.5 # Max L2 distance for vector retrieval
    
    RERANKER_SCORE_THRESHOLD = 0.0  # Min score for re-ranker to consider relevant
    FINAL_K = 5                    # Final number of chunks to send to LLM

config = Config()
