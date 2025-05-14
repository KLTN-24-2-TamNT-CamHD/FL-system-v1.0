import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN

def preprocess_creditcard_dataset(df):
    """
    Preprocess the Credit Card Fraud dataset from Kaggle.
    
    Args:
        df: Raw DataFrame from the creditcard.csv file
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert the Class column to int if it's not already
    df['Class'] = df['Class'].astype(int)
    
    # Create a copy of original Time for potential time-based splits
    df['original_time'] = df['Time']
    
    # Create temporal features from Time (seconds elapsed)
    # Convert seconds to hours of day (assuming Time starts at beginning of day)
    df['hour_of_day'] = (df['Time'] / 3600) % 24
    
    # Create a day indicator (assuming the data spans 2 days)
    df['day'] = (df['Time'] / (3600 * 24)).astype(int)
    
    # Create a copy of original Amount for reference
    df['original_amount'] = df['Amount']
    
    # Log-transform the Amount feature (often helpful for monetary values)
    df['Amount'] = np.log1p(df['Amount'])
    
    # Standardize Amount (PCA features are already normalized)
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    # Drop the original Time column
    df = df.drop('Time', axis=1)
    
    # Rename the target column to be consistent with your previous code
    df = df.rename(columns={'Class': 'is_fraud'})
    
    return df

def generate_time_based_splits(df, num_clients):
    """
    Split data based on transaction time (most realistic for fraud).
    
    Args:
        df: Preprocessed DataFrame
        num_clients: Number of clients to split data for
        
    Returns:
        List of DataFrames, one for each client
    """
    # Sort by time
    df = df.sort_values('original_time')
    
    # Split into chunks
    client_dfs = np.array_split(df, num_clients)
    
    for i, client_df in enumerate(client_dfs):
        print(f"Client {i+1}: Time range {client_df['original_time'].min():.0f} to {client_df['original_time'].max():.0f}")
        print(f"Client {i+1}: Fraud rate: {client_df['is_fraud'].mean() * 100:.3f}%")
    
    return client_dfs

def generate_amount_based_splits(df, num_clients):
    """
    Split data based on transaction amount.
    
    Args:
        df: Preprocessed DataFrame
        num_clients: Number of clients to split data for
        
    Returns:
        List of DataFrames, one for each client
    """
    # Create transaction amount bins
    df['amount_bin'] = pd.qcut(df['original_amount'], num_clients, labels=False)
    
    # Create client dataframes
    client_dfs = []
    for i in range(num_clients):
        client_df = df[df['amount_bin'] == i].copy()
        client_df = client_df.drop('amount_bin', axis=1)
        client_dfs.append(client_df)
        
        print(f"Client {i+1}: Amount range ${client_df['original_amount'].min():.2f} to ${client_df['original_amount'].max():.2f}")
        print(f"Client {i+1}: Fraud rate: {client_df['is_fraud'].mean() * 100:.3f}%")
    
    return client_dfs

def generate_pca_feature_splits(df, num_clients):
    """
    Split based on PCA feature patterns - creates most challenging non-IID distribution.
    
    Args:
        df: Preprocessed DataFrame
        num_clients: Number of clients to split data for
        
    Returns:
        List of DataFrames, one for each client
    """
    # Use V1 and V2 (typically the most significant PCA components) for clustering
    # Select key features for clustering
    features = ['V1', 'V2', 'V3', 'V4']
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clients, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features])
    
    # Create client dataframes
    client_dfs = []
    for i in range(num_clients):
        client_df = df[df['cluster'] == i].copy()
        client_df = client_df.drop('cluster', axis=1)
        client_dfs.append(client_df)
        
        print(f"Client {i+1}: {len(client_df)} transactions")
        print(f"Client {i+1}: Fraud rate: {client_df['is_fraud'].mean() * 100:.3f}%")
    
    return client_dfs

def create_creditcard_visualizations(df, client_dfs, split_type, output_dir):
    """
    Create visualizations of the data distribution.
    
    Args:
        df: Original preprocessed DataFrame
        client_dfs: List of client DataFrames
        split_type: Type of split used
        output_dir: Directory to save visualizations
    """
    # Create directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Class distribution in full dataset
    plt.figure(figsize=(10, 6))
    sns.countplot(x='is_fraud', data=df)
    plt.title('Class Distribution in Full Dataset')
    plt.xlabel('Fraud (0 = No, 1 = Yes)')
    plt.ylabel('Number of Transactions')
    plt.savefig(f"{viz_dir}/class_distribution.png")
    plt.close()
    
    # 2. Class distribution by client
    plt.figure(figsize=(14, 8))
    for i, client_df in enumerate(client_dfs):
        plt.subplot(1, len(client_dfs), i+1)
        client_class_counts = client_df['is_fraud'].value_counts()
        plt.pie(client_class_counts, labels=['Normal', 'Fraud'] if client_class_counts.index[0] == 0 else ['Fraud', 'Normal'],
                autopct='%1.1f%%')
        plt.title(f'Client {i+1}')
    plt.suptitle(f'Class Distribution by Client ({split_type} split)')
    plt.savefig(f"{viz_dir}/class_distribution_by_client.png")
    plt.close()
    
    # 3. Feature distributions
    if split_type == "pca_based":
        # Plot V1 vs V2 with client coloring
        plt.figure(figsize=(12, 10))
        for i, client_df in enumerate(client_dfs):
            plt.scatter(client_df['V1'], client_df['V2'], alpha=0.5, label=f'Client {i+1}')
        plt.title('PCA Feature Distribution (V1 vs V2) by Client')
        plt.xlabel('V1')
        plt.ylabel('V2')
        plt.legend()
        plt.savefig(f"{viz_dir}/pca_features_by_client.png")
        plt.close()
        
        # Plot V1 vs V2 with fraud coloring
        plt.figure(figsize=(12, 10))
        colors = {0: 'blue', 1: 'red'}
        for fraud_value in [0, 1]:
            subset = df[df['is_fraud'] == fraud_value]
            plt.scatter(subset['V1'], subset['V2'], 
                       c=colors[fraud_value], 
                       alpha=0.5, 
                       label=f'{"Fraud" if fraud_value==1 else "Normal"}')
        plt.title('PCA Feature Distribution (V1 vs V2) by Class')
        plt.xlabel('V1')
        plt.ylabel('V2')
        plt.legend()
        plt.savefig(f"{viz_dir}/pca_features_by_class.png")
        plt.close()
    
    elif split_type == "amount_based":
        # Plot amount distribution by client
        plt.figure(figsize=(14, 8))
        for i, client_df in enumerate(client_dfs):
            plt.hist(client_df['original_amount'], alpha=0.5, label=f'Client {i+1}', bins=30)
        plt.title('Transaction Amount Distribution by Client')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f"{viz_dir}/amount_distribution_by_client.png")
        plt.close()
    
    elif split_type == "time_based":
        # Plot time distribution by client
        plt.figure(figsize=(14, 8))
        for i, client_df in enumerate(client_dfs):
            plt.hist(client_df['original_time'], alpha=0.5, label=f'Client {i+1}', bins=30)
        plt.title('Transaction Time Distribution by Client')
        plt.xlabel('Time (seconds from start)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f"{viz_dir}/time_distribution_by_client.png")
        plt.close()

def apply_varying_smote(client_id, X_train, y_train):
    """
    Apply different SMOTE strategies based on client ID.
    
    Args:
        client_id: The identifier for the client (e.g., "client-1")
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Resampled X_train and y_train
    """
    print(f"Before resampling - {client_id} class distribution: {Counter(y_train)}")
    
    # Different strategies for different clients
    if client_id == "client-1":
        # Client 1: Light oversampling
        ratio = 0.05  # 5% fraud
        print(f"{client_id}: SMOTE with ratio {ratio}")
        smote = SMOTE(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elif client_id == "client-2":
        # Client 2: Moderate oversampling
        ratio = 0.1  # 10% fraud
        print(f"{client_id}: SMOTE with ratio {ratio}")
        smote = SMOTE(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elif client_id == "client-3":
        # Client 3: Heavy oversampling (nearly balanced)
        ratio = 0.25  # 25% fraud
        print(f"{client_id}: SMOTE with ratio {ratio}")
        smote = SMOTE(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    else:
        # Other clients: Medium oversampling
        ratio = 0.15  # 15% fraud
        print(f"{client_id}: SMOTE with ratio {ratio}")
        smote = SMOTE(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After resampling - {client_id} class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def save_client_data_to_file(client_data, output_dir):
    """
    Save client data to CSV files with header information.
    
    Args:
        client_data: Dictionary containing client data and metadata
        output_dir: Directory to save files
    """
    client_id = client_data["client_id"]
    train_df = client_data["train"]
    test_df = client_data["test"]
    split_type = client_data["split_type"]
    
    # Create header with metadata
    header = [
        f"# Dataset: Credit Card Fraud Detection (Kaggle)",
        f"# Split type: {split_type}",
        f"# Client ID: {client_id}"
    ]
    
    # Add specific details based on split type
    if split_type == "time_based" and "time_min" in client_data:
        header.append(f"# Time range: {client_data['time_min']:.0f} to {client_data['time_max']:.0f}")
    elif split_type == "amount_based" and "amount_min" in client_data:
        header.append(f"# Amount range: ${client_data['amount_min']:.2f} to ${client_data['amount_max']:.2f}")
    
    # Add class distribution info
    header.append(f"# Train fraud ratio: {client_data['fraud_ratio_train'] * 100:.4f}%")
    header.append(f"# Test fraud ratio: {client_data['fraud_ratio_test'] * 100:.4f}%")
    
    # Add sample counts
    header.append(f"# Train samples: {client_data['train_size']}")
    header.append(f"# Test samples: {client_data['test_size']}")
    
    header.append("")  # Empty line before data
    
    # Save train and test files with header
    train_file = f"{output_dir}/{client_id}_train.txt"
    test_file = f"{output_dir}/{client_id}_test.txt"
    
    # Save train data
    with open(train_file, 'w') as f:
        # Write header
        f.write('\n'.join(header))
        # Write data
        train_df.to_csv(f, index=False)
    
    # Save test data
    with open(test_file, 'w') as f:
        # Write header
        f.write('\n'.join(header))
        # Write data
        test_df.to_csv(f, index=False)

def create_summary_file(client_splits, df, split_type, balance_method, output_dir):
    """
    Create a summary file with information about the data splits.
    
    Args:
        client_splits: Dictionary of client data
        df: Original DataFrame
        split_type: Type of split used
        balance_method: Method used for balancing classes
        output_dir: Directory to save the summary file
    """
    with open(f"{output_dir}/dataset_summary.txt", 'w') as f:
        f.write(f"Credit Card Fraud Detection Dataset - {split_type} Split\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Original fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.4f}%)\n\n")
        
        if balance_method == "smote":
            f.write(f"Balance method: Varying SMOTE (different for each client)\n\n")
        elif balance_method == "adasyn":
            f.write(f"Balance method: Varying ADASYN (different for each client)\n\n")
        else:
            f.write(f"Balance method: {balance_method}\n\n")
        
        f.write("Client Summaries:\n")
        for client_id, data in client_splits.items():
            f.write(f"\n{client_id}:\n")
            
            # Add specific details based on split type
            if split_type == "time_based" and "time_min" in data:
                f.write(f"  - Time range: {data['time_min']:.0f} to {data['time_max']:.0f}\n")
            elif split_type == "amount_based" and "amount_min" in data:
                f.write(f"  - Amount range: ${data['amount_min']:.2f} to ${data['amount_max']:.2f}\n")
            
            # Add class distribution info
            f.write(f"  - Train fraud ratio: {data['fraud_ratio_train'] * 100:.4f}%\n")
            f.write(f"  - Test fraud ratio: {data['fraud_ratio_test'] * 100:.4f}%\n")
            
            # Add sample counts
            f.write(f"  - Train samples: {data['train_size']}\n")
            f.write(f"  - Test samples: {data['test_size']}\n")
            
            # Add fraud counts
            f.write(f"  - Train fraud cases: {int(data['train']['is_fraud'].sum())}\n")
            f.write(f"  - Test fraud cases: {int(data['test']['is_fraud'].sum())}\n")

def generate_balanced_splits_creditcard(
    data_file,
    num_clients=3,
    train_size=30000,
    test_size=5000,
    split_type="time_based",
    balance_method="smote",
    visualization=True,
    output_dir="."
):
    """
    Generate balanced credit card fraud dataset splits for federated learning.
    
    Args:
        data_file: Path to the creditcard.csv file
        num_clients: Number of clients to split data for
        train_size: Target number of training samples per client
        test_size: Target number of test samples per client
        split_type: "time_based", "amount_based", or "pca_based"
        balance_method: "smote", "adasyn", "weight", or "none"
        visualization: Whether to create visualization of the data distribution
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with client data splits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    fraud_count = df['Class'].sum()
    fraud_percentage = fraud_count / len(df) * 100
    print(f"Class distribution - Fraud: {fraud_count} ({fraud_percentage:.4f}%), Non-Fraud: {len(df) - fraud_count}")
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    df = preprocess_creditcard_dataset(df)
    
    # Generate splits based on chosen strategy
    client_dfs = []
    if split_type == "time_based":
        client_dfs = generate_time_based_splits(df, num_clients)
    elif split_type == "amount_based":
        client_dfs = generate_amount_based_splits(df, num_clients)
    elif split_type == "pca_based":
        client_dfs = generate_pca_feature_splits(df, num_clients)
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    # Create visualizations if requested
    if visualization:
        create_creditcard_visualizations(df, client_dfs, split_type, output_dir)
    
    # Process each client's data
    client_splits = {}
    
    for i, client_df in enumerate(client_dfs):
        client_id = f"client-{i+1}"
        
        # Split into train/test with stratification
        train_df, test_df = train_test_split(
            client_df, 
            test_size=min(test_size, len(client_df) * 0.3),
            train_size=min(train_size, len(client_df) * 0.7),
            stratify=client_df['is_fraud'],
            random_state=42
        )
        
        # Apply balance method if requested
        if balance_method == "smote" and len(train_df) > 0:
            # Apply client-specific SMOTE strategies
            X_train = train_df.drop('is_fraud', axis=1)
            y_train = train_df['is_fraud']
            
            # Apply specific SMOTE ratio based on client ID
            X_resampled, y_resampled = apply_varying_smote(client_id, X_train, y_train)
            
            # Recombine into a DataFrame
            train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
            train_df['is_fraud'] = y_resampled
        
        elif balance_method == "adasyn" and len(train_df) > 0:
            # ADASYN focuses more on difficult-to-learn examples
            X_train = train_df.drop('is_fraud', axis=1)
            y_train = train_df['is_fraud']
            
            # Determine ratio based on client ID
            if client_id == "client-1":
                ratio = 0.05
            elif client_id == "client-2":
                ratio = 0.10
            else:
                ratio = 0.15
                
            print(f"{client_id}: Applying ADASYN with ratio {ratio}")
            print(f"Before ADASYN - Class distribution: {Counter(y_train)}")
            
            # Apply ADASYN
            adasyn = ADASYN(sampling_strategy=ratio, random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
            
            print(f"After ADASYN - Class distribution: {Counter(y_resampled)}")
            
            # Recombine into a DataFrame
            train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
            train_df['is_fraud'] = y_resampled
        
        # Collect client data info
        client_data = {
            "train": train_df,
            "test": test_df,
            "split_type": split_type,
            "client_id": client_id,
            "fraud_ratio_train": train_df['is_fraud'].mean(),
            "fraud_ratio_test": test_df['is_fraud'].mean(),
            "train_size": len(train_df),
            "test_size": len(test_df)
        }
        
        # Add specific details based on split type
        if split_type == "time_based":
            client_data["time_min"] = client_df['original_time'].min()
            client_data["time_max"] = client_df['original_time'].max()
        elif split_type == "amount_based":
            client_data["amount_min"] = client_df['original_amount'].min()
            client_data["amount_max"] = client_df['original_amount'].max()
        
        client_splits[client_id] = client_data
        
        # Save train and test files
        save_client_data_to_file(client_data, output_dir)
    
    # Create a summary file
    create_summary_file(client_splits, df, split_type, balance_method, output_dir)
    
    return client_splits

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate balanced credit card fraud dataset splits for federated learning")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to the creditcard.csv file")
    parser.add_argument("--split-type", type=str, default="pca_based", 
                        choices=["time_based", "amount_based", "pca_based"],
                        help="Type of dataset split to generate")
    parser.add_argument("--num-clients", type=int, default=3,
                        help="Number of clients to generate data for")
    parser.add_argument("--train-size", type=int, default=30000,
                        help="Target number of training samples per client")
    parser.add_argument("--test-size", type=int, default=5000,
                        help="Target number of test samples per client")
    parser.add_argument("--balance-method", type=str, default="smote",
                        choices=["smote", "adasyn", "weight", "none"],
                        help="Method to balance the classes")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Directory to save the output files")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of the data distribution")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate the dataset splits
    print(f"Generating {args.split_type} splits for {args.num_clients} clients...")
    if args.balance_method in ["smote", "adasyn"]:
        print(f"Using varying {args.balance_method.upper()} ratios tailored to each client")
    
    start_time = datetime.now()
    splits = generate_balanced_splits_creditcard(
        data_file=args.data_file,
        num_clients=args.num_clients,
        train_size=args.train_size,
        test_size=args.test_size,
        split_type=args.split_type,
        balance_method=args.balance_method,
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
        print(f"Visualizations saved to {args.output_dir}/visualizations")