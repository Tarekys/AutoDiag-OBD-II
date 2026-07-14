# Architecture Overview

## Runtime Components

- `src/car_fault_prediction/api/main.py`: FastAPI application and `/predict/` endpoint.
- `src/car_fault_prediction/services/prediction_service.py`: prediction orchestration, SMS throttling, and SQLite persistence.
- `src/car_fault_prediction/utils/preprocessing.py`: missing-value handling and categorical encoding.
- `src/car_fault_prediction/ui/dashboard_app.py`: main Streamlit dashboard.
- `src/car_fault_prediction/ui/chart_generator.py`: chart generation utilities.
- `src/car_fault_prediction/ui/pages/elm327_analytics.py`: analytics page.

## Supporting Assets

- `models/artifacts/`: trained XGBoost model, encoders, and feature column metadata.
- `database/sample.db`: legacy database artifact kept for reference.
- `static/chatbot_llm/`: static chatbot HTML prototype and Dockerfile.
- `notebooks/xgboost_model.ipynb`: original training notebook.

## Compatibility Layer

The `API/` and `UI/` folders now contain thin wrapper files so existing launch points still work while the implementation lives in a maintainable `src/` package layout.
