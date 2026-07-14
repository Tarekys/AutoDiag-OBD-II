from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
DATABASE_DIR = PROJECT_ROOT / "database"
STATIC_DIR = PROJECT_ROOT / "static"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

load_dotenv(PROJECT_ROOT / ".env")

MODEL_PATH = MODELS_DIR / "car_fault_classifier.json"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
SAMPLE_DATABASE_PATH = DATABASE_DIR / "sample.db"

# Preserve the existing runtime behavior by defaulting to the original
# relative database filename when no explicit path is configured.
RUNTIME_DATABASE_PATH = Path(os.getenv("OBD_DATABASE_PATH", "obd_data.db"))

FASTAPI_URL = os.getenv("FASTAPI_URL", "********************")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "twilio_account_sid")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "twilio_auth_token")
TWILIO_MESSAGING_SERVICE_SID = os.getenv(
    "TWILIO_MESSAGING_SERVICE_SID",
    "twilio_messaging_service_sid",
)
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "phone")
