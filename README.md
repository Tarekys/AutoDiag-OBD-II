# Real-Time Car Fault Prediction System

## Overview

Real-Time Car Fault Prediction System, a robust machine learning-based solution designed to predict vehicle faults using real-time data such as engine RPM, tire pressure, and other diagnostic metrics. This project leverages advanced algorithms and a user-friendly interface dashboard to provide actionable insights for automotive maintenance.

## Features

- **Real-Time Prediction**: Utilizes live data streams to forecast potential car faults.
- **Machine Learning Models**: Employs XGBoost, DNN for regression to enhance prediction accuracy.
- **API Integration**: Built with FastAPI for seamless data handling and deployment.
- **Interactive UI**: Features a Streamlit-based interface with visualizations like line graphs for tire pressure.
- **Data Processing**: Supports CSV file inputs for testing and validation.

## Project Structure

```
├── API.py              # FastAPI endpoint for predictions
├── Procfile            # The configuration for deployment
├── app.py              # Main Integrated application script
├── car_fault_classifier.json  # Trained model configuration
├── encoders.pkl        # Encoded data for preprocessing
├── feature_columns.pkl # Feature set metadata
├── predictor.py        # Prediction logic
├── requirements.txt    # Project dependencies
├── utilize.py          # Helpers functions
```

## Installation

### Prerequisites
- Python 3.12+
- Git

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/real-time-car-fault-prediction.git
   cd real-time-car-fault-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure required data files (`car_fault_classifier.json`, `encoders.pkl`, `feature_columns.pkl`) are in the project directory.

## Usage

### Running the Application
1. Start the FastAPI server:
   ```bash
   python API.py
   ```
2. Launch the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
3. Access the UI at `http://localhost...` and the API at `http://localhost...`.

### Testing
- Use `testing data.csv` to simulate real-time data and verify predictions.
- Modify `predictor.py` to adjust model parameters or add new features.

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m "Description of changes"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or support, please open an issue on the GitHub repository or contact with `tarekys9939@gamil.com`.

## Acknowledgments

- Thanks to the open-source community for tools like XGBoost, FastAPI, and Streamlit.
