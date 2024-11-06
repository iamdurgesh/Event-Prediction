import os
import torch
import json
import argparse
import logging
from data_preprocessing import preprocess_data
from global_context_extraction import create_global_context_dataset
from train import main as train_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_and_filter_data(input_path, output_path):
    data = pd.read_csv(input_path)
    preprocessed_data = preprocess_data(data)
    preprocessed_data.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")

def create_context_dataset(preprocessed_data_path, context_output_path):
    high_impact_event_columns = [
        'event_vehicle_detection', 'event_pedestrian_detection', 'event_traffic_sign', 
        'event_red_traffic_light', 'lane_change', 'left_lane_change', 'right_lane_change', 
        'event_global_path', 'event_intersection'
    ]
    create_global_context_dataset(pd.read_csv(preprocessed_data_path), high_impact_event_columns, save_format="json")
    logger.info(f"Global context dataset saved to {context_output_path}")

def run_training():
    train_model()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    logger.info("Step 1: Preprocessing Data...")
    preprocess_and_filter_data(args.raw_data_path, args.preprocessed_data_path)
    
    logger.info("Step 2: Creating Global Context Dataset...")
    create_context_dataset(args.preprocessed_data_path, args.context_dataset_path)
    
    logger.info("Step 3: Training Model...")
    run_training()
    logger.info("Project completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the event prediction project pipeline.")
    parser.add_argument('--raw_data_path', default='data/initial_data_format.csv', help="Path to raw data file.")
    parser.add_argument('--preprocessed_data_path', default='data/preprocessed_data.csv', help="Path to save preprocessed data.")
    parser.add_argument('--context_dataset_path', default='data/final_global_context_dataset.json', help="Path to save context dataset.")
    args = parser.parse_args()
    
    main(args)
