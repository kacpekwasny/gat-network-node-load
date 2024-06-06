# GNN Network Management System

This project uses Graph Neural Networks (GNN) to manage and optimize network parameters.

## Requirements

Ensure you have the following dependencies installed. You can install them using `pip`:

``` pip install -r requirements.txt ```

## Project Structure

- `generate_network_data.py`: Script to generate synthetic network data.
- `preprocess.py`: Script to preprocess the generated network data.
- `check_processed_data.py`: Script to verify the processed data.
- `train_gat.py`: Script to train the Graph Attention Network (GAT) model.

## Instructions

### Step 1: Data Generation

Run the `generate_network_data.py` script to generate the synthetic network data.

```python generate_network_data.py```

This script will:

- Generate synthetic network data and save it to a file.

### Step 2: Data Preprocessing

Run the `preprocess.py` script to preprocess the generated network data.

```python preprocess.py```

This script will:

- Preprocess the data and save it to a file (e.g., `processed_data.pt`).

### Step 3: Check Processed Data

Run the `check_processed_data.py` script to verify the processed data.

```python check_processed_data.py```

This script will:

- Load and check the processed data for correctness.

### Step 4: Model Training

Run the `train_gat.py` script to train the GAT model.

```python train_gat.py```

This script will:

- Load the preprocessed data.
- Initialize and train the GAT model.
- Generate and save loss curves and other training metrics.
