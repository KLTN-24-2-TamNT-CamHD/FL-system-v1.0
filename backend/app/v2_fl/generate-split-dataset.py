import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import argparse

def generate_diabetes_splits(
    num_clients=3, 
    split_type="overlapping", 
    overlap_percentage=30,
    visualization=False,
    output_dir="."
):
    """
    Generate Diabetes dataset splits for federated learning with different strategies.
    
    Args:
        num_clients: Number of clients to split data for
        split_type: Splitting strategy - "iid", "non-iid", "overlapping", or "stratified"
        overlap_percentage: Percentage of data overlap between clients (for overlapping split)
        visualization: Whether to create visualization of the data distribution
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with client data splits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Diabetes dataset
    diabetes = load_diabetes()
    
    # Extract feature names
    feature_names = diabetes.feature_names
    
    # Create a DataFrame for easier handling
    df = pd.DataFrame(diabetes.data, columns=feature_names)
    df['target'] = diabetes.target
    
    # Add a binned version of the target for stratification
    df['target_bin'] = pd.qcut(df['target'], 10, labels=False)
    
    # Standardize features
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    
    # Save full dataset info for header
    column_info = {
        'feature_names': feature_names,
        'target_name': 'target',
        'description': diabetes.DESCR.split('\n')[0:5],  # Keep only first 5 lines of description
        'num_samples': len(df),
        'num_features': len(feature_names)
    }
    
    client_splits = {}
    
    if split_type == "iid":
        # IID split: random sampling
        # First shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Then split into num_clients equal parts
        splits = np.array_split(df, num_clients)
        
        for i, split_df in enumerate(splits):
            # Further split into train/test
            train_df, test_df = train_test_split(split_df, test_size=0.2, random_state=42)
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "iid",
                "client_id": f"client-{i+1}"
            }
            
    elif split_type == "non-iid":
        # Non-IID split: sort by target value and split (no overlap)
        # Sort by target value
        df = df.sort_values(by='target').reset_index(drop=True)
        
        # Split into num_clients equal parts (each client gets a different range of diabetes progression)
        splits = np.array_split(df, num_clients)
        
        for i, split_df in enumerate(splits):
            # Further split into train/test
            train_df, test_df = train_test_split(split_df, test_size=0.2, random_state=42)
            
            # Calculate target range for this client
            target_min = float(split_df['target'].min())
            target_max = float(split_df['target'].max())
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "non-iid",
                "target_range": [target_min, target_max],
                "client_id": f"client-{i+1}"
            }
            
    elif split_type == "overlapping":
        # Overlapping split: Each client gets some unique data and some shared data
        # Sort by target value
        df = df.sort_values(by='target').reset_index(drop=True)
        
        # Calculate the length of each client's dataset
        base_size = len(df) // num_clients
        
        # Calculate the overlap size
        overlap_size = int(base_size * (overlap_percentage / 100))
        effective_size = base_size + overlap_size
        
        # For visualization
        client_ranges = []
        
        for i in range(num_clients):
            # Calculate the start and end indices for this client's data
            # Each client gets its own section plus some data from adjacent sections
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
            
            # Further split into train/test
            train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42)
            
            # Record the target range for this client
            target_min = float(client_df['target'].min())
            target_max = float(client_df['target'].max())
            client_ranges.append((start_idx, end_idx, target_min, target_max))
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "overlapping",
                "overlap_percentage": overlap_percentage,
                "target_range": [target_min, target_max],
                "client_id": f"client-{i+1}"
            }
            
        if visualization:
            # Create a visualization of how the data is split
            plt.figure(figsize=(12, 6))
            
            # Plot the full dataset's target distribution
            plt.hist(df['target'], bins=30, alpha=0.3, color='gray', label='Full Dataset')
            
            # Plot each client's range
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for i, (start, end, tmin, tmax) in enumerate(client_ranges):
                client_data = df.iloc[start:end]
                plt.hist(client_data['target'], bins=20, alpha=0.5, 
                         color=colors[i % len(colors)], 
                         label=f'Client {i+1} ({tmin:.2f}-{tmax:.2f})')
            
            plt.title(f'Diabetes Progression Distribution - {split_type.capitalize()} Split with {overlap_percentage}% Overlap')
            plt.xlabel('Disease Progression (Target)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/diabetes_split_visualization.png")
            plt.close()
    
    elif split_type == "stratified":
        # Stratified split: Each client gets a representative sample across all progression values
        # but with different distributions
        
        # Group by target bins
        bins = df['target_bin'].unique()
        bin_groups = {bin_val: df[df['target_bin'] == bin_val] for bin_val in bins}
        
        client_dfs = [pd.DataFrame() for _ in range(num_clients)]
        
        # Create different distributions for each client but ensure they all have some 
        # data from every progression range
        for bin_val, bin_df in bin_groups.items():
            # Create different sampling proportions for each client
            if num_clients == 3:
                # Predefined proportions for 3 clients to create meaningful differences
                if bin_val < 3:  # Lower progression bins
                    proportions = [0.6, 0.3, 0.1]  # Client 1 gets more low progression
                elif bin_val < 7:  # Middle progression bins
                    proportions = [0.2, 0.6, 0.2]  # Client 2 gets more middle progression
                else:  # Higher progression bins
                    proportions = [0.1, 0.3, 0.6]  # Client 3 gets more high progression
            else:
                # For any number of clients, create a gradient of proportions
                center = num_clients // 2
                distances = [abs(i - (bin_val * num_clients) / 10) for i in range(num_clients)]
                total = sum(distances)
                # Invert so closer gets higher proportion
                proportions = [1 - (d / total) for d in distances]
                # Normalize to sum to 1
                proportions = [p / sum(proportions) for p in proportions]
            
            # Split this bin's data according to the proportions
            bin_df_shuffled = bin_df.sample(frac=1, random_state=int(42 + bin_val))
            
            start_idx = 0
            for i in range(num_clients):
                # Calculate how many samples this client gets from this bin
                samples = int(len(bin_df) * proportions[i])
                # Ensure last client gets remaining samples to avoid rounding issues
                if i == num_clients - 1:
                    end_idx = len(bin_df)
                else:
                    end_idx = start_idx + samples
                
                # Add these samples to the client's dataframe
                client_dfs[i] = pd.concat([client_dfs[i], bin_df_shuffled.iloc[start_idx:end_idx]])
                
                start_idx = end_idx
        
        # Process each client's data
        for i, client_df in enumerate(client_dfs):
            # Shuffle the data
            client_df = client_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split into train/test
            train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42)
            
            # Record target range
            target_min = float(client_df['target'].min())
            target_max = float(client_df['target'].max())
            
            # Calculate the progression distribution profile
            progress_profile = client_df.groupby('target_bin').size().to_dict()
            
            client_splits[f"client-{i+1}"] = {
                "train": train_df,
                "test": test_df,
                "split_type": "stratified",
                "target_range": [target_min, target_max],
                "progress_profile": progress_profile,
                "client_id": f"client-{i+1}"
            }
        
        if visualization:
            # Create a visualization of the progression distributions
            plt.figure(figsize=(12, 6))
            
            bin_centers = [(i + 0.5) for i in range(10)]  # Centers of the bins
            bin_labels = [f"{i}" for i in range(10)]  # Labels for the bins
            
            # Plot each client's distribution
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for i, client_df in enumerate(client_dfs):
                dist = client_df.groupby('target_bin').size() / len(client_df)
                plt.bar([x + i*0.2 for x in bin_centers], 
                        dist.values, 
                        width=0.2, 
                        alpha=0.7,
                        color=colors[i % len(colors)], 
                        label=f'Client {i+1}')
            
            plt.title('Disease Progression Distribution by Client (Stratified Split)')
            plt.xlabel('Progression Range (Low to High)')
            plt.ylabel('Proportion of Client Data')
            plt.xticks(bin_centers, bin_labels)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/stratified_split_visualization.png")
            plt.close()
    
    # Save each client's data to CSV files
    for client_id, data in client_splits.items():
        # Create header with metadata
        header = [
            f"# Dataset: Diabetes",
            f"# Split type: {data['split_type']}",
            f"# Client ID: {client_id}"
        ]
        
        if "overlap_percentage" in data:
            header.append(f"# Overlap percentage: {data['overlap_percentage']}%")
        
        header.extend([
            f"# Features: {', '.join(feature_names)}",
            f"# Target: target"
        ])
        
        if "target_range" in data:
            header.append(f"# Target range: {data['target_range'][0]:.4f} to {data['target_range'][1]:.4f}")
        
        header.append(f"# Train samples: {len(data['train'])}")
        header.append(f"# Test samples: {len(data['test'])}")
        header.append("")  # Empty line before data
        
        # Save train and test files with header
        train_file = f"{output_dir}/{client_id}_train.txt"
        test_file = f"{output_dir}/{client_id}_test.txt"
        
        # Remove the target_bin column for the output files
        train_df = data['train'].drop('target_bin', axis=1) if 'target_bin' in data['train'].columns else data['train']
        test_df = data['test'].drop('target_bin', axis=1) if 'target_bin' in data['test'].columns else data['test']
        
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
    
    # Create a summary file
    with open(f"{output_dir}/dataset_summary.txt", 'w') as f:
        f.write(f"Diabetes Dataset - {split_type.capitalize()} Split\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {', '.join(feature_names)}\n\n")
        
        f.write("Client Summaries:\n")
        for client_id, data in client_splits.items():
            f.write(f"\n{client_id}:\n")
            f.write(f"  - Split type: {data['split_type']}\n")
            if "overlap_percentage" in data:
                f.write(f"  - Overlap percentage: {data['overlap_percentage']}%\n")
            if "target_range" in data:
                f.write(f"  - Target range: {data['target_range'][0]:.4f} to {data['target_range'][1]:.4f}\n")
            f.write(f"  - Train samples: {len(data['train'])}\n")
            f.write(f"  - Test samples: {len(data['test'])}\n")
            if "progress_profile" in data:
                f.write(f"  - Progression distribution: {data['progress_profile']}\n")
    
    print(f"Target value range in dataset: {df['target'].min():.2f} to {df['target'].max():.2f}")
    return client_splits

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Diabetes dataset splits for federated learning")
    parser.add_argument("--split-type", type=str, default="overlapping", 
                        choices=["iid", "non-iid", "overlapping", "stratified"],
                        help="Type of dataset split to generate")
    parser.add_argument("--num-clients", type=int, default=3,
                        help="Number of clients to generate data for")
    parser.add_argument("--overlap", type=int, default=30,
                        help="Percentage of overlap between clients (for overlapping split)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save the output files")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of the data distribution")
    
    args = parser.parse_args()
    
    # Generate the dataset splits
    print(f"Generating {args.split_type} dataset splits for {args.num_clients} clients...")
    if args.split_type == "overlapping":
        print(f"Using {args.overlap}% overlap between clients")
    
    splits = generate_diabetes_splits(
        num_clients=args.num_clients,
        split_type=args.split_type,
        overlap_percentage=args.overlap,
        visualization=args.visualize,
        output_dir=args.output_dir
    )
    
    print(f"Done! Files created in {args.output_dir}:")
    for client_id in splits.keys():
        print(f"  {client_id}_train.txt")
        print(f"  {client_id}_test.txt")
    
    if args.visualize:
        print(f"Visualization saved to {args.output_dir}")
