import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
import zipfile
from io import StringIO
from imblearn.over_sampling import SMOTE
from collections import Counter

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
    
    # One-hot encode categorical columns
    for col in categorical_columns:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    # Standardize numeric columns (except target)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols.remove('is_fraud')  # Don't standardize the target
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def apply_smote(X_train, y_train, sampling_strategy=0.1):
    """
    Apply SMOTE oversampling to balance the classes.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Ratio of minority to majority class after resampling
        
    Returns:
        Resampled X_train and y_train
    """
    print(f"Before SMOTE - Class distribution: {Counter(y_train)}")
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def generate_balanced_splits(
    data_file,
    num_clients=3,
    train_size=50000,
    test_size=10000,
    split_type="by_feature",
    balance_method="smote",
    smote_ratio=0.1,
    visualization=False,
    output_dir="."
):
    """
    Generate balanced credit card fraud detection dataset splits with fixed sizes.
    
    Args:
        data_file: Path to the fraud detection dataset CSV file
        num_clients: Number of clients to split data for
        train_size: Number of training samples per client
        test_size: Number of test samples per client
        split_type: "by_feature" or "by_amount"
        balance_method: "smote", "weight", or "none"
        smote_ratio: Target ratio for minority class after SMOTE
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
    
    # Check if we have enough data
    total_samples_needed = num_clients * (train_size + test_size)
    if len(df) < total_samples_needed:
        print(f"Warning: Dataset has {len(df)} samples, but {total_samples_needed} are needed.")
        print("Using all available data and scaling down the requested sample sizes proportionally.")
        
        # Scale down sample sizes
        scaling_factor = len(df) / total_samples_needed
        train_size = int(train_size * scaling_factor)
        test_size = int(test_size * scaling_factor)
        print(f"Adjusted train_size: {train_size}, test_size: {test_size} per client")
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    df = preprocess_fraud_dataset(df)
    
    print(f"Dataset after preprocessing: {len(df)} rows and {len(df.columns)} columns")
    fraud_count = df['is_fraud'].sum()
    fraud_percentage = fraud_count / len(df) * 100
    print(f"Class distribution - Fraud: {fraud_count} ({fraud_percentage:.2f}%), Non-Fraud: {len(df) - fraud_count}")
    
    # Get feature names for documentation
    target_column = 'is_fraud'
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Create visualizations if requested
    if visualization:
        # Class distribution
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
    
    if split_type == "by_amount":
        # Split based on transaction amounts
        print("Creating splits based on transaction amounts...")
        
        # Sort by transaction amount
        df = df.sort_values(by='amt').reset_index(drop=True)
        
        # Calculate total samples per client
        samples_per_client = train_size + test_size
        
        # Get the fraud indices
        fraud_indices = df[df['is_fraud'] == 1].index.tolist()
        non_fraud_indices = df[df['is_fraud'] == 0].index.tolist()
        
        # Distribute fraud cases among clients to ensure each gets some fraud cases
        fraud_per_client = len(fraud_indices) // num_clients
        
        for i in range(num_clients):
            # Calculate start and end indices for this client's range
            start_idx = i * (len(df) // num_clients)
            end_idx = (i + 1) * (len(df) // num_clients) if i < num_clients - 1 else len(df)
            
            # Get data for this client's range
            client_df = df.iloc[start_idx:end_idx].copy()
            
            # Calculate amount range and fraud ratio
            amount_min = float(client_df['amt'].min())
            amount_max = float(client_df['amt'].max())
            fraud_ratio = client_df['is_fraud'].mean() * 100
            
            # Save the range info
            client_data = {
                "split_type": "by_amount",
                "amount_range": [amount_min, amount_max],
                "fraud_ratio": fraud_ratio,
                "client_id": f"client-{i+1}"
            }
            
            # Further split into train/test
            # Be sure both sets have some fraud cases
            fraud_df = client_df[client_df['is_fraud'] == 1]
            non_fraud_df = client_df[client_df['is_fraud'] == 0]
            
            # Split fraud cases 80/20
            if len(fraud_df) > 0:
                fraud_train, fraud_test = train_test_split(fraud_df, test_size=0.2, random_state=42)
            else:
                fraud_train = pd.DataFrame(columns=client_df.columns)
                fraud_test = pd.DataFrame(columns=client_df.columns)
            
            # Calculate how many non-fraud cases we need
            non_fraud_train_size = min(train_size - len(fraud_train), len(non_fraud_df) - len(fraud_test))
            non_fraud_train_size = max(0, non_fraud_train_size)  # Make sure it's not negative
            
            # Split non-fraud cases
            if non_fraud_train_size > 0 and len(non_fraud_df) > 0:
                non_fraud_train, remaining = train_test_split(
                    non_fraud_df, 
                    train_size=non_fraud_train_size, 
                    random_state=42
                )
                non_fraud_test = remaining.iloc[:min(test_size - len(fraud_test), len(remaining))]
            else:
                non_fraud_train = pd.DataFrame(columns=client_df.columns)
                non_fraud_test = pd.DataFrame(columns=client_df.columns)
            
            # Combine the data
            train_df = pd.concat([fraud_train, non_fraud_train])
            test_df = pd.concat([fraud_test, non_fraud_test])
            
            # Shuffle
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Check actual sizes
            client_data["actual_train_size"] = len(train_df)
            client_data["actual_test_size"] = len(test_df)
            client_data["actual_fraud_count_train"] = train_df['is_fraud'].sum()
            client_data["actual_fraud_count_test"] = test_df['is_fraud'].sum()
            
            # Apply SMOTE if requested
            if balance_method == "smote" and len(train_df) > 0:
                X_train = train_df.drop('is_fraud', axis=1)
                y_train = train_df['is_fraud']
                
                X_resampled, y_resampled = apply_smote(X_train, y_train, smote_ratio)
                
                # Recombine into a DataFrame
                train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
                train_df['is_fraud'] = y_resampled
                
                client_data["resampled_train_size"] = len(train_df)
                client_data["resampled_fraud_count_train"] = train_df['is_fraud'].sum()
            
            # Add the final train/test sets to the client data
            client_data["train"] = train_df
            client_data["test"] = test_df
            
            # Add to the splits
            client_splits[f"client-{i+1}"] = client_data
            
            print(f"Client {i+1}:")
            print(f"  - Amount range: ${amount_min:.2f} to ${amount_max:.2f}")
            print(f"  - Original fraud ratio: {fraud_ratio:.2f}%")
            print(f"  - Train set size: {len(train_df)}, Test set size: {len(test_df)}")
            print(f"  - Fraud in train: {train_df['is_fraud'].sum()} ({train_df['is_fraud'].mean()*100:.2f}%)")
            
    else:  # split_type == "by_feature"
        # Split based on transaction features to create distinct domains
        print("Creating splits based on transaction features...")
        
        # Select a feature to split on (e.g., transaction_month)
        split_feature = 'category_entertainment' if 'category_entertainment' in df.columns else 'transaction_month'
        
        # Create a composite feature for better separation
        df['split_score'] = df[split_feature]
        if 'amt' in df.columns:
            # Normalize and add transaction amount to create more distinct domains
            df['split_score'] = df['split_score'] + (df['amt'] - df['amt'].min()) / (df['amt'].max() - df['amt'].min())
        
        # Sort by the composite feature
        df = df.sort_values(by='split_score').reset_index(drop=True)
        
        # Create non-overlapping splits
        splits = np.array_split(df, num_clients)
        
        for i, split_df in enumerate(splits):
            # Calculate feature range and fraud ratio
            feature_min = float(split_df['split_score'].min())
            feature_max = float(split_df['split_score'].max())
            fraud_ratio = split_df['is_fraud'].mean() * 100
            
            # Save the range info
            client_data = {
                "split_type": "by_feature",
                "feature_range": [feature_min, feature_max],
                "fraud_ratio": fraud_ratio,
                "client_id": f"client-{i+1}"
            }
            
            # Further split into train/test
            fraud_df = split_df[split_df['is_fraud'] == 1]
            non_fraud_df = split_df[split_df['is_fraud'] == 0]
            
            # Calculate maximum possible sizes while maintaining stratification
            max_train_size = min(train_size, len(split_df) * 0.8)
            max_test_size = min(test_size, len(split_df) * 0.2)
            
            # Split while preserving class distribution
            train_df, test_df = train_test_split(
                split_df, 
                test_size=max_test_size/(max_train_size+max_test_size),
                stratify=split_df['is_fraud'],
                random_state=42
            )
            
            # Limit to requested sizes if needed
            if len(train_df) > train_size:
                train_df = train_df.sample(train_size, random_state=42)
            if len(test_df) > test_size:
                test_df = test_df.sample(test_size, random_state=42)
            
            # Drop the split_score column
            train_df = train_df.drop('split_score', axis=1)
            test_df = test_df.drop('split_score', axis=1)
            
            # Check actual sizes
            client_data["actual_train_size"] = len(train_df)
            client_data["actual_test_size"] = len(test_df)
            client_data["actual_fraud_count_train"] = train_df['is_fraud'].sum()
            client_data["actual_fraud_count_test"] = test_df['is_fraud'].sum()
            
            # Apply SMOTE if requested
            if balance_method == "smote" and len(train_df) > 0:
                X_train = train_df.drop('is_fraud', axis=1)
                y_train = train_df['is_fraud']
                
                X_resampled, y_resampled = apply_smote(X_train, y_train, smote_ratio)
                
                # Recombine into a DataFrame
                train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
                train_df['is_fraud'] = y_resampled
                
                client_data["resampled_train_size"] = len(train_df)
                client_data["resampled_fraud_count_train"] = train_df['is_fraud'].sum()
            
            # Add the final train/test sets to the client data
            client_data["train"] = train_df
            client_data["test"] = test_df
            
            # Add to the splits
            client_splits[f"client-{i+1}"] = client_data
            
            print(f"Client {i+1}:")
            print(f"  - Feature range: {feature_min:.2f} to {feature_max:.2f}")
            print(f"  - Original fraud ratio: {fraud_ratio:.2f}%")
            print(f"  - Train set size: {len(train_df)}, Test set size: {len(test_df)}")
            print(f"  - Fraud in train: {train_df['is_fraud'].sum()} ({train_df['is_fraud'].mean()*100:.2f}%)")
    
    # Save each client's data to CSV files
    for client_id, data in client_splits.items():
        # Create header with metadata
        header = [
            f"# Dataset: Credit Card Fraud Detection",
            f"# Split type: {data['split_type']}",
            f"# Client ID: {client_id}"
        ]
        
        if "amount_range" in data:
            header.append(f"# Amount range: ${data['amount_range'][0]:.2f} to ${data['amount_range'][1]:.2f}")
        
        if "feature_range" in data:
            header.append(f"# Feature range: {data['feature_range'][0]:.2f} to {data['feature_range'][1]:.2f}")
        
        header.append(f"# Original fraud ratio: {data['fraud_ratio']:.2f}%")
        
        train_fraud_ratio = data['train']['is_fraud'].mean() * 100
        header.append(f"# Train fraud ratio: {train_fraud_ratio:.2f}%")
        
        test_fraud_ratio = data['test']['is_fraud'].mean() * 100
        header.append(f"# Test fraud ratio: {test_fraud_ratio:.2f}%")
        
        header.append(f"# Features: {', '.join(feature_columns)}")
        header.append(f"# Target: {target_column} (0 = legitimate, 1 = fraud)")
        header.append(f"# Train samples: {len(data['train'])}")
        header.append(f"# Test samples: {len(data['test'])}")
        
        if balance_method == "smote":
            header.append(f"# Balance method: SMOTE with ratio {smote_ratio}")
            
        elif balance_method == "weight":
            # Calculate class weights
            weight_ratio = 1 / data['train']['is_fraud'].mean() / 2
            header.append(f"# Balance method: Class weights (negative=1.0, positive={weight_ratio:.2f})")
            
        header.append("")  # Empty line before data
        
        # Add weights for weighted loss functions
        if balance_method == "weight":
            weights_info = {
                'class_0_weight': 1.0,
                'class_1_weight': 1 / data['train']['is_fraud'].mean() / 2
            }
            header.append(f"# Class weights: {weights_info}")
            header.append("")
        
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
        f.write(f"Credit Card Fraud Detection Dataset - {split_type} Split\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {len(feature_columns)}\n")
        f.write(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)\n\n")
        
        if balance_method == "smote":
            f.write(f"Balance method: SMOTE with ratio {smote_ratio}\n\n")
        elif balance_method == "weight":
            f.write(f"Balance method: Class weights\n\n")
        else:
            f.write(f"Balance method: None (original class distribution)\n\n")
        
        f.write("Client Summaries:\n")
        for client_id, data in client_splits.items():
            f.write(f"\n{client_id}:\n")
            
            if "amount_range" in data:
                f.write(f"  - Amount range: ${data['amount_range'][0]:.2f} to ${data['amount_range'][1]:.2f}\n")
            
            if "feature_range" in data:
                f.write(f"  - Feature range: {data['feature_range'][0]:.2f} to {data['feature_range'][1]:.2f}\n")
            
            f.write(f"  - Original fraud ratio: {data['fraud_ratio']:.2f}%\n")
            
            train_fraud_ratio = data['train']['is_fraud'].mean() * 100
            f.write(f"  - Train fraud ratio: {train_fraud_ratio:.2f}%\n")
            
            test_fraud_ratio = data['test']['is_fraud'].mean() * 100
            f.write(f"  - Test fraud ratio: {test_fraud_ratio:.2f}%\n")
            
            f.write(f"  - Train samples: {len(data['train'])}\n")
            f.write(f"  - Test samples: {len(data['test'])}\n")
            
            f.write(f"  - Train fraud cases: {data['train']['is_fraud'].sum()}\n")
            f.write(f"  - Test fraud cases: {data['test']['is_fraud'].sum()}\n")
    
    return client_splits

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate balanced fraud detection dataset splits for federated learning")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to the fraud detection dataset CSV or ZIP file")
    parser.add_argument("--split-type", type=str, default="by_feature", 
                        choices=["by_amount", "by_feature"],
                        help="Type of dataset split to generate")
    parser.add_argument("--num-clients", type=int, default=3,
                        help="Number of clients to generate data for")
    parser.add_argument("--train-size", type=int, default=50000,
                        help="Number of training samples per client")
    parser.add_argument("--test-size", type=int, default=10000,
                        help="Number of test samples per client")
    parser.add_argument("--balance-method", type=str, default="smote",
                        choices=["smote", "weight", "none"],
                        help="Method to balance the classes")
    parser.add_argument("--smote-ratio", type=float, default=0.1,
                        help="Target ratio for SMOTE (minority:majority)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save the output files")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of the data distribution")
    
    args = parser.parse_args()
    
    # Generate the dataset splits
    print(f"Generating {args.split_type} splits for {args.num_clients} clients...")
    print(f"Target sizes - Train: {args.train_size}, Test: {args.test_size} per client")
    
    start_time = datetime.now()
    splits = generate_balanced_splits(
        data_file=args.data_file,
        num_clients=args.num_clients,
        train_size=args.train_size,
        test_size=args.test_size,
        split_type=args.split_type,
        balance_method=args.balance_method,
        smote_ratio=args.smote_ratio,
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
