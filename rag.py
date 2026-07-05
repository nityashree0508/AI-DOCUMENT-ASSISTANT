import os
from dotenv import load_dotenv

loaded = load_dotenv()



GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_DIR = "data"

VECTOR_DB_PATH = "vector_store/faiss"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 5
