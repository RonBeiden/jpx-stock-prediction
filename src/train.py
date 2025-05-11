import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from data_processing.data_loader import JPXDataLoader
from models.lstm_model import LSTMModel
from evaluation.metrics import evaluate_predictions

def train_model(
    data_dir: str,
    model_dir: str,
    start_date: str = None,
    end_date: str = None,
    sequence_length: int = 10,
    validation_split: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32
):
    """
    Train the model and save results.
    
    Args:
        data_dir: Directory containing input data
        model_dir: Directory to save model and results
        start_date: Start date for training data
        end_date: End date for training data
        sequence_length: Length of sequences for LSTM
        validation_split: Fraction of data for validation
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    # Create output directory
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    data_loader = JPXDataLoader(data_dir)
    df = data_loader.prepare_training_data(start_date, end_date)
    
    # Initialize and train model
    model = LSTMModel(sequence_length=sequence_length)
    X, y = model.preprocess(df)
    
    print(f"Training data shape: {X.shape}")
    model.train(
        X, y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Make predictions
    predictions = model.predict(X)
    
    # Prepare evaluation DataFrame
    eval_df = df.iloc[sequence_length:].copy()
    eval_df["Prediction"] = predictions
    
    # Calculate metrics
    mean_corr, daily_corrs = evaluate_predictions(eval_df)
    print(f"\nMean correlation: {mean_corr:.4f}")
    
    # Save results
    model.save_model(model_dir / "model.h5")
    daily_corrs.to_csv(model_dir / "daily_correlations.csv", index=False)
    
    # Save sample predictions
    eval_df[["Date", "SecuritiesCode", "Target", "Prediction"]].to_csv(
        model_dir / "predictions.csv",
        index=False
    )

def main():
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument("--data_dir", required=True, help="Input data directory")
    parser.add_argument("--model_dir", required=True, help="Output model directory")
    parser.add_argument("--start_date", help="Start date for training")
    parser.add_argument("--end_date", help="End date for training")
    parser.add_argument("--sequence_length", type=int, default=10,
                      help="Sequence length for LSTM")
    parser.add_argument("--validation_split", type=float, default=0.2,
                      help="Validation data fraction")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    
    args = parser.parse_args()
    
    train_model(
        args.data_dir,
        args.model_dir,
        args.start_date,
        args.end_date,
        args.sequence_length,
        args.validation_split,
        args.epochs,
        args.batch_size
    )

if __name__ == "__main__":
    main() 