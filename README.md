# JPX Tokyo Stock Exchange Prediction

This project aims to predict future returns of Japanese stocks listed on the Tokyo Stock Exchange as part of the [JPX Tokyo Stock Exchange Prediction Challenge](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/overview).

## Project Structure

```
jpx_stock_prediction/
├── data/               # Data files and preprocessing scripts
├── notebooks/         # Jupyter notebooks for EDA and experimentation
├── models/           # Saved model files and model-specific code
├── src/             # Source code for the project
└── requirements.txt  # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Components

1. **Data Processing** (`src/data_processing/`)
   - Feature engineering
   - Data cleaning and preprocessing
   - Target variable creation

2. **Models** (`src/models/`)
   - Traditional ML models (LightGBM, XGBoost, CatBoost)
   - Time Series models (ARIMA, VAR)
   - Deep Learning models (LSTM)

3. **Evaluation** (`src/evaluation/`)
   - Time-based cross-validation
   - Model performance metrics
   - Prediction ranking

## Usage

1. Run EDA notebooks in `notebooks/` to understand the data
2. Execute preprocessing scripts in `src/data_processing/`
3. Train models using scripts in `src/models/`
4. Generate predictions and prepare submission

## Evaluation Metric

The competition uses a Weighted Correlation metric for the top 200 predictions ranked by confidence values.

## License

This project is for educational purposes and participation in the Kaggle competition. 