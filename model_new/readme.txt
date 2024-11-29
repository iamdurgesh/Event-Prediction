## Saving and Loading the Model
1. **Save the Trained Model**: After training, the model is saved as `model_weights_epoch10.pth`.
2. **Load the Pre-trained Model**: Use the `load_model` flag in `main.py` to load a pre-trained model for evaluation or fine-tuning.

### Fine-Tuning
- Modify the `epochs` parameter and continue training from the loaded model.



# Instruction Script for Event Prediction Model: Code Responsibilities and Execution Steps
Overview
This script provides details on the responsibilities of each code file/module and step-by-step instructions for training, testing, and evaluating the event prediction model. Instructions are provided for both normal and masked evaluations.

Code Responsibilities
1. main.py
Responsibilities:
Entry point for the model training and evaluation pipeline.
Configures datasets, loads the model, and orchestrates training and evaluation processes.
Provides toggles for normal and masked evaluations using evaluate_only and excluded_events.
Key Variables:
evaluate_only: Set to True for evaluation-only mode and False for training + evaluation.
excluded_events: Specifies events to be excluded in masked evaluation mode.
config: Dictionary containing model configurations such as batch size, embedding size, and vocabulary size.

2. model.py
Responsibilities:
Defines the TransformerSeq2SeqModel architecture.
Includes embedding layers, positional encoding, and transformer encoder/decoder modules.
Final output layer maps token representations to the vocabulary size for predictions.
Key Points:
embedding.weight maps tokens to dense vectors.
fc_out maps token representations to vocabulary predictions.

3. train.py
Responsibilities:
Implements training and validation loops.
Utilizes cross-entropy loss for sequence-to-sequence predictions.
Incorporates gradient clipping to stabilize training.
Functions:
train_model: Performs training over specified epochs and records loss.
validate_model: Validates the model on the validation set during training.

4. evaluation.py or masked_evaluation.py
Responsibilities:
Contains evaluation functions for generating metrics like accuracy, precision, recall, and F1-score.
masked_evaluation.py: Performs evaluation considering masked events and excluded IDs.

5. utils.py
Responsibilities:
Implements the collate_fn function to prepare batches for the DataLoader.
Handles the exclusion of specified events and masking logic.
6. dataset.py

Responsibilities:
Implements the EventDataset class for loading and preprocessing datasets.
Converts raw CSV files into tokenized sequences suitable for training.

7. train_balanced.csv, val.csv, test.csv
Responsibilities:
Training, validation, and test datasets in CSV format.
Each file contains sequences with columns for events and a cycle column.

8. To split the dataset from scratch, run repartitioned_data.py, and it will repartition the dataset
9. To Downsample the train.csv, test.csv, val.csv file, you can run the downsampled_data.py file

Execution Steps
1. Setup
Ensure the necessary datasets (train.csv, val.csv, test.csv) are in the working directory.
Install the required Python packages using pip install -r requirements.txt.
2. For Normal Evaluation (No Masking)
Set Configuration in main.py:

Ensure evaluate_only = False to train the model.
Comment out or remove the excluded_events logic to include all events during training and testing.
python
Code kopieren
evaluate_only = False
excluded_events = None
Run the Code:

bash
Code kopieren
python main.py
Output:

Training will begin, and loss curves will be saved to the results directory.
After training, evaluation metrics (accuracy, precision, recall, F1-score) will be printed.
3. For Masked Evaluation (Exclude Specific Events)
Set Configuration in main.py:

Set evaluate_only = True to skip training and evaluate an existing model.
Define excluded_events to specify events to exclude (e.g., event_task_start, event_task_end, event_start_calc_local_context).
python
Code kopieren
evaluate_only = True
excluded_events = {"event_task_start", "event_task_end", "event_start_calc_local_context"}
Adjust collate_fn in utils.py:

Ensure collate_fn applies the mask for excluded events during both training and evaluation.
Run the Code:

bash
Code kopieren
python main.py
Output:

Model will be evaluated on the test dataset, excluding specified events.
Masked evaluation metrics (accuracy, precision, recall, F1-score) and a classification report will be printed.
4. Debugging and Logs
Enable debug logs in train.py, main.py, or collate_fn to inspect the shapes of inputs, masks, and targets during data loading and processing.
Use assertions in collate_fn to ensure no empty sequences are passed to the model.
Key Configuration Notes
Model Configurations (config):

Ensure vocab_size matches the number of events after applying excluded_events.
Update embed_size, num_heads, and num_layers to match the model architecture used during training.
Model Checkpoints:

Always train and save the model using the same configuration (vocab_size, embedding size, etc.) as used during evaluation.
Example Outputs
Normal Evaluation:

yaml
Code kopieren
Evaluation Metrics:
Accuracy: 92.34%
Precision: 91.80%
Recall: 92.34%
F1 Score: 92.01%
Masked Evaluation:

yaml
Code kopieren
Evaluation Metrics (Masked):
Accuracy: 85.12%
Precision: 84.90%
Recall: 85.12%
F1 Score: 84.80%
Classification Report (Masked):
...