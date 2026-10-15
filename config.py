import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Dataset
RAW_FILE = "Sample - Superstore.csv"
DATE_COL = "Order Date"
TARGET_COL = "Sales"

# PostgreSQL
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "sales_db")
DB_USER = os.getenv("DB_USER", "sales_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_URL = (
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Warehouse table names
FACT_TABLE = "fact_sales"
DIM_PRODUCT = "dim_product"
DIM_CUSTOMER = "dim_customer"
DIM_LOCATION = "dim_location"
DIM_DATE = "dim_date"

# Model
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_DEPTH = 5
LEARNING_RATE = 0.05
FORECAST_HORIZON = 90

# API
API_HOST = "0.0.0.0"
API_PORT = 8001
