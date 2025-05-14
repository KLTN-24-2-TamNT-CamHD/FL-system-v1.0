import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
import zipfile
from io import StringIO

def preprocess_fraud_dataset(df):
    """
    Preprocess the credit card fraud detection dataset.
    
    Args:
        df: Raw DataFrame from the fraud detection dataset
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date and time columns to datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    
    # Extract useful features from datetime
    df['transaction_day'] = df['trans_date_trans_time'].dt.day
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
    df['transaction_month'] = df['trans_date_trans_time'].dt.month
    df['transaction_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    
    # Drop columns that are not useful for prediction or contain sensitive information
    columns_to_drop = [
        'trans_date_trans_time',  # Already extracted features
        'merchant',               # Too many categories, can cause overfitting
        'first',                  # Personal information
        'last',                   # Personal information
        'street',                 # Personal information
        'city',                   # Location already captured in city_pop and other features
        'job',                    # Not strongly predictive
        'dob',                    # Age is more useful
        'trans_num',              # Just an ID
        'unix_time',              # Already captured in other time features
        'zip',                    # Location already captured in city_pop
        'lat', 'long',            # Location already captured in more structured features
    ]
    
    df = df.drop(columns=columns_to_drop)
    
    # Handle categorical columns
    categorical_columns = ['category', 'gender', 'state']
    
    # Fill missing values (if any)
    df = df.fillna({
        'category': 'unknown',
        'gender': 'unknown',
        'state': 'unknown'
    })
    
    # Convert fraud column to binary (0 or 1)
    df['is_fraud'] = df['is_fraud'].astype(int)
    
    # Calculate age from date of birth (if dob is not dropped)
    if 'dob' in df.columns:
        df['age'] = (pd.to_datetime('now') - pd.to_datetime(df['dob'])).dt.days // 365
        df = df.drop(columns=['dob'])
    
    return df

def generate_fraud_detection_splits(
    data_file,
    num_clients=3, 
    split_type="stratified", 
    overlap_percentage=30,
    max_samples=None,
    visualization=False,
    output_dir="."
):
    """
    Generate credit card fraud detection dataset splits for federated learning.
    
    Args:
        data_file: Path to the fraud detection dataset CSV file
        num_clients: Number of clients to split data for
        split_type: Splitting strategy - "iid", "non-iid", "overlapping", "stratified", "temporal"
        overlap_percentage: Percentage of data overlap between clients (for overlapping split)
        max_samples: Maximum number of samples to use (useful for testing)
        visualization: Whether to create visualization of the data distribution
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with client data splits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {data_file}...")
    
    # Check if it's a zip file
    if data_file.endswith('.zip'):
        with zipfile.ZipFile(data_file, 'r') as zip_ref:
            # Find the CSV file inside the zip
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found inside the ZIP file")
            
            # Read the first CSV file
            with zip_ref.open(csv_files[0]) as f:
                content = f.read().decode('utf-8')
                df = pd.read_csv(StringIO(content))
    else:
        # Read directly from CSV
        df = pd.read_csv(data_file)
    
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Limit the number of samples if specified
    if max_samples and max_samples < len(df):
        print(f"Limiting to {max_samples} samples for processing")
        df = df.sample(max_samples, random_state=42)
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    df = preprocess_fraud_dataset(df)
    
    print(f"Dataset after preprocessing: {len(df)} rows and {len(df.columns)} columns")
    print(f"Class distribution - Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%), Non-Fraud: {len(df) - df['is_fraud'].sum()}")
    
    # Get feature names for documentation
    target_column = 'is_fraud'
    feature_columns = [col for col in df.columns if col != target_column]
    
    # For visualization and stratification
    if visualization:
        # Create distribution visualizations
        plt.figure(figsize=(10, 6))
        plt.bar(['Legitimate', 'Fraudulent'], 
                [len(df) - df['is_fraud'].sum(), df['is_fraud'].sum()])
        plt.title('Class Distribution in Full Dataset')
        plt.ylabel('Number of Transactions')
        plt.savefig(f"{output_dir}/fraud_class_distribution.png")
        plt.close()
        
        # Transaction amount distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='amt', hue='is_fraud', bins=50, log_scale=True)
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Transaction Amount (log scale)')
        plt.ylabel('Frequency')
        plt.savefig(f"{output_dir}/fraud_amount_distribution.png")
        plt.close()
    
    client_splits = {}
    
    if split_type == "iid":
        # IID split: random sampling with preserved class distribution
        print("Creating IID split with preserved class distribution...")
        
        # Separate fraud and non-fraud transactions
        fraud_df = df[df['is_fraud'] == 1]
        legitimate_df = df[df['is_fraud'] == 0]
        
        # Split each into num_clients parts
        fraud_splits = np.array_split(fraud_df.sample(frac=1, random_state=42), num_clients)
        legitimate_splits = np.array_split(legitimate_df.sample(frac=1, random_state=42), num_clients)
        
        for i in range(num_clients):
            # Combine fraud and legitimate for this client
            client_df = pd.concat([fraud_splits[i], legitimate_splits[i]])
            
            # Shuffle
            client_df = client_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split into train and test
            train_df, test_df = train_test_split(client_df, test_size=0.2, 
                                               stratify=client_df['is_fraud'],
                                               random_state=42)
            
            # Calculate fraud ratio
            fraud_ratio = client_df['is_fraud'].mean() * 100
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "iid",
                "fraud_ratio": fraud_ratio,
                "client_id": f"client-{i+1}"
            }
            
            print(f"Client {i+1} - Total: {len(client_df)}, Fraud: {client_df['is_fraud'].sum()} ({fraud_ratio:.2f}%)")
            
    elif split_type == "non-iid":
        # Non-IID split based on transaction amounts
        print("Creating non-IID split based on transaction amounts...")
        
        # Sort by transaction amount
        df = df.sort_values(by='amt').reset_index(drop=True)
        
        # Split into num_clients equal parts
        splits = np.array_split(df, num_clients)
        
        for i, split_df in enumerate(splits):
            # Further split into train/test with stratification
            train_df, test_df = train_test_split(split_df, test_size=0.2, 
                                               stratify=split_df['is_fraud'],
                                               random_state=42)
            
            # Calculate amount range and fraud ratio
            amount_min = float(split_df['amt'].min())
            amount_max = float(split_df['amt'].max())
            fraud_ratio = split_df['is_fraud'].mean() * 100
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "non-iid",
                "amount_range": [amount_min, amount_max],
                "fraud_ratio": fraud_ratio,
                "client_id": f"client-{i+1}"
            }
            
            print(f"Client {i+1} - Amount range: ${amount_min:.2f} to ${amount_max:.2f}, " + 
                  f"Fraud: {split_df['is_fraud'].sum()} ({fraud_ratio:.2f}%)")
            
    elif split_type == "overlapping":
        # Overlapping split based on transaction amounts with overlap
        print(f"Creating overlapping split with {overlap_percentage}% overlap...")
        
        # Sort by transaction amount
        df = df.sort_values(by='amt').reset_index(drop=True)
        
        # Calculate the base size for each client
        base_size = len(df) // num_clients
        
        # Calculate the overlap size
        overlap_size = int(base_size * (overlap_percentage / 100))
        
        # For visualization
        client_ranges = []
        
        for i in range(num_clients):
            # Calculate the start and end indices for this client's data
            if i == 0:
                # First client gets its section plus overlap from next section
                start_idx = 0
                end_idx = base_size + overlap_size
            elif i == num_clients - 1:
                # Last client gets its section plus overlap from previous section
                start_idx = i * base_size - overlap_size
                end_idx = len(df)
            else:
                # Middle clients get overlap from both sides
                start_idx = i * base_size - overlap_size // 2
                end_idx = (i + 1) * base_size + overlap_size // 2
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(df), end_idx)
            
            # Get data for this client
            client_df = df.iloc[start_idx:end_idx].copy()
            
            # Further split into train/test with stratification
            train_df, test_df = train_test_split(client_df, test_size=0.2, 
                                               stratify=client_df['is_fraud'],
                                               random_state=42)
            
            # Calculate amount range and fraud ratio
            amount_min = float(client_df['amt'].min())
            amount_max = float(client_df['amt'].max())
            fraud_ratio = client_df['is_fraud'].mean() * 100
            client_ranges.append((start_idx, end_idx, amount_min, amount_max))
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "overlapping",
                "overlap_percentage": overlap_percentage,
                "amount_range": [amount_min, amount_max],
                "fraud_ratio": fraud_ratio,
                "client_id": f"client-{i+1}"
            }
            
            print(f"Client {i+1} - Amount range: ${amount_min:.2f} to ${amount_max:.2f}, " + 
                  f"Fraud: {client_df['is_fraud'].sum()} ({fraud_ratio:.2f}%)")
        
        if visualization:
            # Visualize the overlapping splits
            plt.figure(figsize=(12, 6))
            
            # Plot the full dataset's amount distribution
            plt.hist(df['amt'], bins=50, alpha=0.3, color='gray', label='Full Dataset')
            
            # Plot each client's range
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for i, (start, end, amin, amax) in enumerate(client_ranges):
                client_data = df.iloc[start:end]
                plt.hist(client_data['amt'], bins=30, alpha=0.5, 
                         color=colors[i % len(colors)], 
                         label=f'Client {i+1} (${amin:.2f}-${amax:.2f})')
            
            plt.title(f'Transaction Amount Distribution - {split_type.capitalize()} Split with {overlap_percentage}% Overlap')
            plt.xlabel('Transaction Amount')
            plt.ylabel('Frequency')
            plt.xscale('log')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/overlapping_split_visualization.png")
            plt.close()
            
    elif split_type == "stratified":
        # Stratified split: Each client gets data from different amount ranges but with
        # proportionally similar fraud ratios
        print("Creating stratified split with balanced fraud distribution...")
        
        # Create transaction amount bins
        num_bins = 10
        df['amount_bin'] = pd.qcut(df['amt'], num_bins, labels=False)
        
        # Separate fraud and legitimate transactions within each bin
        bin_fraud_groups = {}
        bin_legitimate_groups = {}
        
        for bin_val in range(num_bins):
            bin_df = df[df['amount_bin'] == bin_val]
            bin_fraud_groups[bin_val] = bin_df[bin_df['is_fraud'] == 1]
            bin_legitimate_groups[bin_val] = bin_df[bin_df['is_fraud'] == 0]
        
        # Initialize client dataframes
        client_dfs = [pd.DataFrame() for _ in range(num_clients)]
        
        # Distribute fraud transactions first to ensure balance
        for bin_val, fraud_df in bin_fraud_groups.items():
            # Divide fraud transactions equally among clients
            fraud_splits = np.array_split(fraud_df.sample(frac=1, random_state=42), num_clients)
            
            # Add to each client
            for i in range(num_clients):
                client_dfs[i] = pd.concat([client_dfs[i], fraud_splits[i]])
        
        # Now distribute legitimate transactions to maintain similar class imbalance
        for bin_val, legitimate_df in bin_legitimate_groups.items():
            # Calculate how many legitimate transactions each client should get
            # to maintain a similar fraud ratio
            legitimate_per_client = []
            remaining = len(legitimate_df)
            
            for i in range(num_clients):
                # Get current client fraud count
                client_fraud = client_dfs[i]['is_fraud'].sum()
                
                # Calculate desired legitimate count based on global fraud ratio
                global_ratio = df['is_fraud'].sum() / len(df)
                desired_legitimate = int(client_fraud / global_ratio) - client_fraud
                
                # Adjust for available legitimate transactions
                legitimate_per_client.append(min(desired_legitimate, remaining // (num_clients - i)))
                remaining -= legitimate_per_client[-1]
            
            # Distribute legitimate transactions
            legitimate_shuffled = legitimate_df.sample(frac=1, random_state=42).reset_index(drop=True)
            start_idx = 0
            
            for i in range(num_clients):
                end_idx = start_idx + legitimate_per_client[i]
                client_dfs[i] = pd.concat([client_dfs[i], legitimate_shuffled.iloc[start_idx:end_idx]])
                start_idx = end_idx
        
        # Process each client's data
        for i, client_df in enumerate(client_dfs):
            # Shuffle the data
            client_df = client_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split into train/test with stratification
            train_df, test_df = train_test_split(client_df, test_size=0.2, 
                                               stratify=client_df['is_fraud'],
                                               random_state=42)
            
            # Calculate statistics
            amount_min = float(client_df['amt'].min())
            amount_max = float(client_df['amt'].max())
            fraud_ratio = client_df['is_fraud'].mean() * 100
            amount_distribution = client_df.groupby('amount_bin').size().to_dict()
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df.drop('amount_bin', axis=1),  # Remove bin column
                "test": test_df.drop('amount_bin', axis=1),    # Remove bin column
                "split_type": "stratified",
                "amount_range": [amount_min, amount_max],
                "fraud_ratio": fraud_ratio,
                "client_id": f"client-{i+1}"
            }
            
            print(f"Client {i+1} - Samples: {len(client_df)}, " +
                  f"Fraud: {client_df['is_fraud'].sum()} ({fraud_ratio:.2f}%)")
        
        # Remove the bin column from the dataframe
        df = df.drop('amount_bin', axis=1)
            
    elif split_type == "temporal":
        # Temporal split: Split by transaction time
        print("Creating temporal split based on transaction dates...")
        
        # Make sure we have the date column
        if 'trans_date_trans_time' not in df.columns:
            raise ValueError("Dataset doesn't contain transaction date/time information")
        
        # Sort by transaction date/time
        df = df.sort_values(by='trans_date_trans_time').reset_index(drop=True)
        
        # Split into num_clients sequential time periods
        splits = np.array_split(df, num_clients)
        
        for i, split_df in enumerate(splits):
            # Further split into train/test with stratification
            train_df, test_df = train_test_split(split_df, test_size=0.2, 
                                               stratify=split_df['is_fraud'],
                                               random_state=42)
            
            # Calculate time range and fraud ratio
            time_min = split_df['trans_date_trans_time'].min()
            time_max = split_df['trans_date_trans_time'].max()
            fraud_ratio = split_df['is_fraud'].mean() * 100
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "temporal",
                "time_range": [time_min.strftime('%Y-%m-%d'), time_max.strftime('%Y-%m-%d')],
                "fraud_ratio": fraud_ratio,
                "client_id": f"client-{i+1}"
            }
            
            print(f"Client {i+1} - Time range: {time_min.strftime('%Y-%m-%d')} to {time_max.strftime('%Y-%m-%d')}, " + 
                  f"Fraud: {split_df['is_fraud'].sum()} ({fraud_ratio:.2f}%)")
    
    # Save each client's data to CSV files
    for client_id, data in client_splits.items():
        # Create header with metadata
        header = [
            f"# Dataset: Credit Card Fraud Detection",
            f"# Split type: {data['split_type']}",
            f"# Client ID: {client_id}"
        ]
        
        if "overlap_percentage" in data:
            header.append(f"# Overlap percentage: {data['overlap_percentage']}%")
        
        if "amount_range" in data:
            header.append(f"# Amount range: ${data['amount_range'][0]:.2f} to ${data['amount_range'][1]:.2f}")
        
        if "time_range" in data:
            header.append(f"# Time range: {data['time_range'][0]} to {data['time_range'][1]}")
        
        header.append(f"# Fraud ratio: {data['fraud_ratio']:.2f}%")
        header.append(f"# Features: {', '.join(feature_columns)}")
        header.append(f"# Target: {target_column} (0 = legitimate, 1 = fraud)")
        header.append(f"# Train samples: {len(data['train'])}")
        header.append(f"# Test samples: {len(data['test'])}")
        header.append("")  # Empty line before data
        
        # Save train and test files with header
        train_file = f"{output_dir}/{client_id}_train.txt"
        test_file = f"{output_dir}/{client_id}_test.txt"
        
        # Save train data
        with open(train_file, 'w') as f:
            # Write header
            f.write('\n'.join(header))
            # Write data
            data['train'].to_csv(f, index=False)
        
        # Save test data
        with open(test_file, 'w') as f:
            # Write header
            f.write('\n'.join(header))
            # Write data
            data['test'].to_csv(f, index=False)
    
    # Create a summary file
    with open(f"{output_dir}/dataset_summary.txt", 'w') as f:
        f.write(f"Credit Card Fraud Detection Dataset - {split_type.capitalize()} Split\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {', '.join(feature_columns)}\n")
        f.write(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)\n\n")
        
        f.write("Client Summaries:\n")
        for client_id, data in client_splits.items():
            f.write(f"\n{client_id}:\n")
            f.write(f"  - Split type: {data['split_type']}\n")
            if "overlap_percentage" in data:
                f.write(f"  - Overlap percentage: {data['overlap_percentage']}%\n")
            if "amount_range" in data:
                f.write(f"  - Amount range: ${data['amount_range'][0]:.2f} to ${data['amount_range'][1]:.2f}\n")
            if "time_range" in data:
                f.write(f"  - Time range: {data['time_range'][0]} to {data['time_range'][1]}\n")
            f.write(f"  - Fraud ratio: {data['fraud_ratio']:.2f}%\n")
            f.write(f"  - Train samples: {len(data['train'])}\n")
            f.write(f"  - Test samples: {len(data['test'])}\n")
    
    return client_splits

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Credit Card Fraud Detection dataset splits for federated learning")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to the fraud detection dataset CSV or ZIP file")
    parser.add_argument("--split-type", type=str, default="stratified", 
                        choices=["iid", "non-iid", "overlapping", "stratified", "temporal"],
                        help="Type of dataset split to generate")
    parser.add_argument("--num-clients", type=int, default=3,
                        help="Number of clients to generate data for")
    parser.add_argument("--overlap", type=int, default=30,
                        help="Percentage of overlap between clients (for overlapping split)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to use (for testing)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save the output files")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of the data distribution")
    
    args = parser.parse_args()
    
    # Generate the dataset splits
    print(f"Generating {args.split_type} dataset splits for {args.num_clients} clients...")
    if args.split_type == "overlapping":
        print(f"Using {args.overlap}% overlap between clients")
    
    start_time = datetime.now()
    splits = generate_fraud_detection_splits(
        data_file=args.data_file,
        num_clients=args.num_clients,
        split_type=args.split_type,
        overlap_percentage=args.overlap,
        max_samples=args.max_samples,
        visualization=args.visualize,
        output_dir=args.output_dir
    )
    end_time = datetime.now()
    
    print(f"Processing completed in {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"Files created in {args.output_dir}:")
    for client_id in splits.keys():
        print(f"  {client_id}_train.txt")
        print(f"  {client_id}_test.txt")
    
    if args.visualize:
        print(f"Visualizations saved to {args.output_dir}")