from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Dataset
RAW_FILE         = "Sample - Superstore.csv"
DATE_COL         = "Order Date"
TARGET_COL       = "Sales"

# PostgreSQL
DB_HOST          = "localhost"
DB_PORT          = 5432
DB_NAME          = "sales_db"
DB_USER          = "sales_user"
DB_PASSWORD      = "sales123"
DB_URL           = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Warehouse table names
FACT_TABLE       = "fact_sales"
DIM_PRODUCT      = "dim_product"
DIM_CUSTOMER     = "dim_customer"
DIM_LOCATION     = "dim_location"
DIM_DATE         = "dim_date"

# Model
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
N_ESTIMATORS     = 300
MAX_DEPTH        = 5
LEARNING_RATE    = 0.05
FORECAST_HORIZON = 90

# API
API_HOST         = "0.0.0.0"
API_PORT         = 8001