# Car Fault Prediction System

Production-organized graduation project for vehicle fault prediction using OBD-II data, FastAPI, Streamlit, SQLite, and an XGBoost model.

## Overview

This system simulates or receives vehicle telemetry, preprocesses the input, predicts the most likely fault class with a trained XGBoost classifier, stores prediction results in SQLite, and exposes the output through a FastAPI backend and a Streamlit dashboard.

## Tech Stack

- Python
- FastAPI
- Streamlit
- XGBoost
- scikit-learn
- SQLite
- Plotly
- Twilio

## Project Structure

```text
AutoDiag-OBD-II/
├── API/                         # Backward-compatible API entrypoint and Dockerfile
├── UI/                          # Backward-compatible Streamlit entrypoints and Dockerfile
├── assets/                      # Project screenshots and visual assets
├── database/                    # Database artifacts
├── docs/                        # Architecture and project documentation
├── models/
│   └── artifacts/               # Trained model and preprocessing artifacts
├── notebooks/                   # Training and experimentation notebooks
├── src/
│   └── car_fault_prediction/
│       ├── api/                 # FastAPI implementation
│       ├── config/              # Centralized paths and environment settings
│       ├── services/            # Prediction and persistence orchestration
│       ├── ui/                  # Streamlit dashboard and analytics pages
│       └── utils/               # Shared preprocessing helpers
├── static/
│   └── chatbot_llm/             # Static chatbot prototype assets
├── tests/                       # Smoke tests
├── .env.example                 # Example environment variables
├── requirements.txt
└── README.md
```

## Environment Variables

Copy `.env.example` to `.env` and update values as needed:

- `FASTAPI_URL`
- `OBD_DATABASE_PATH`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_MESSAGING_SERVICE_SID`
- `TWILIO_PHONE_NUMBER`

## Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API from the repository root:

```bash
uvicorn --app-dir API API:app --reload
```

3. Run the Streamlit dashboard from the repository root:

```bash
streamlit run UI/Fault-Dashboard.py
```

## Docker

Build from the repository root so Docker can access the shared `src/`, `models/`, and `database/` folders.

API:

```bash
docker build -f API/Dockerfile -t obd-api .
docker run -p 8000:8000 obd-api
```

UI:

```bash
docker build -f UI/Dockerfile -t obd-ui .
docker run -p 10000:10000 obd-ui
```

Static chatbot prototype:

```bash
docker build -f static/chatbot_llm/Dockerfile -t obd-chatbot .
docker run -p 8080:80 obd-chatbot
```

## Notes

- The business logic, API behavior, model artifacts, and prediction labels are preserved.
- Legacy launch paths in `API/` and `UI/` remain available through compatibility wrappers.
- The training notebook is retained under `notebooks/` for reference.
