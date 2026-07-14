from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_fault_prediction.ui.pages.elm327_analytics import display_analytics_page  # noqa: E402


if __name__ == "__main__":
    display_analytics_page()
