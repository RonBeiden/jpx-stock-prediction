import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from data_processing.data_loader import JPXDataLoader
from models.lstm_model import LSTMModel

def prepare_submission(
    data_dir: str,
    model_path: str,
    output_file: str,
    sequence_length: int = 10
):
    """
    Prepare competition submission file.
    
    Args:
        data_dir: Directory containing input data
        model_path: Path to trained model file
        output_file: Path to save submission file
        sequence_length: Length of sequences for LSTM
    """
    # Load test data
    data_loader = JPXDataLoader(data_dir)
    test_df = data_loader.prepare_training_data()
    
    # Load model
    model = LSTMModel(sequence_length=sequence_length)
    model.load_model(model_path)
    
    # Preprocess and get predictions
    X, _ = model.preprocess(test_df)
    predictions = model.predict(X)
    
    # Prepare submission DataFrame
    submission_df = test_df.iloc[sequence_length:].copy()
    submission_df["Prediction"] = predictions
    
    # For each date, rank stocks by prediction confidence
    ranked_predictions = []
    
    for date in submission_df["Date"].unique():
        day_df = submission_df[submission_df["Date"] == date].copy()
        
        # Rank by absolute prediction value (confidence)
        day_df["Rank"] = day_df["Prediction"].abs().rank(ascending=False)
        
        # Keep top 200 predictions
        day_df = day_df[day_df["Rank"] <= 200]
        ranked_predictions.append(day_df)
    
    # Combine all predictions
    final_df = pd.concat(ranked_predictions)
    
    # Prepare submission format
    submission = final_df[["Date", "SecuritiesCode", "Rank"]]
    submission["Rank"] = submission["Rank"].astype(int)
    
    # Save submission
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved to {output_file}")
    
    # Print sample
    print("\nSample of submission file:")
    print(submission.head())

def main():
    parser = argparse.ArgumentParser(description="Generate competition submission")
    parser.add_argument("--data_dir", required=True,
                      help="Directory containing input data")
    parser.add_argument("--model_path", required=True,
                      help="Path to trained model file")
    parser.add_argument("--output_file", required=True,
                      help="Path to save submission file")
    parser.add_argument("--sequence_length", type=int, default=10,
                      help="Sequence length for LSTM")
    
    args = parser.parse_args()
    
    prepare_submission(
        args.data_dir,
        args.model_path,
        args.output_file,
        args.sequence_length
    )

if __name__ == "__main__":
    main() 