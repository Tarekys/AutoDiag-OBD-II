from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_package_imports():
    from car_fault_prediction.api.main import app
    from car_fault_prediction.services.prediction_service import preprocess_and_predict_from_df
    from car_fault_prediction.ui.chart_generator import ChartGenerator
    from car_fault_prediction.ui.dashboard_app import main
    from car_fault_prediction.ui.pages.elm327_analytics import display_analytics_page
    from car_fault_prediction.utils.preprocessing import encode_categorical_columns, fill_missing

    assert app is not None
    assert callable(preprocess_and_predict_from_df)
    assert ChartGenerator is not None
    assert callable(main)
    assert callable(display_analytics_page)
    assert callable(fill_missing)
    assert callable(encode_categorical_columns)
