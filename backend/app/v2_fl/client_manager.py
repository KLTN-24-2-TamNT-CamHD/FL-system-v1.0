"""
Client management tool for federated learning with blockchain authentication.
"""

import os
import json
import argparse
import sys
import csv
from typing import Dict, List, Optional, Any
from tabulate import tabulate
from datetime import datetime

from blockchain_connector import BlockchainConnector

def load_blockchain_connector(
    ganache_url: str = "http:/192.168.1.146:7545",
    contract_address: Optional[str] = None,
    private_key: Optional[str] = None
) -> BlockchainConnector:
    """
    Load the blockchain connector.
    
    Args:
        ganache_url: Ganache blockchain URL
        contract_address: Address of deployed EnhancedModelRegistry contract
        private_key: Private key for blockchain transactions
        
    Returns:
        BlockchainConnector instance
    """
    # Check if contract address is provided or stored in file
    if contract_address is None:
        try:
            with open("contract_address.txt", "r") as f:
                contract_address = f.read().strip()
                print(f"Loaded contract address from file: {contract_address}")
        except FileNotFoundError:
            print("Error: No contract address provided or found in file")
            print("Please specify a contract address with --contract-address")
            sys.exit(1)
    
    # Initialize blockchain connector
    try:
        blockchain_connector = BlockchainConnector(
            ganache_url=ganache_url,
            contract_address=contract_address,
            private_key=private_key
        )
        print(f"Connected to blockchain at {ganache_url}")
        print(f"Contract address: {contract_address}")
        return blockchain_connector
    except Exception as e:
        print(f"Error connecting to blockchain: {e}")
        sys.exit(1)

def list_clients(blockchain_connector: BlockchainConnector, show_details: bool = False) -> None:
    """
    List all authorized clients.
    
    Args:
        blockchain_connector: Blockchain connector instance
        show_details: Whether to show detailed stats for each client
    """
    try:
        # Get all authorized clients
        clients = blockchain_connector.get_all_authorized_clients()
        
        if not clients:
            print("No authorized clients found")
            return
        
        print(f"Found {len(clients)} authorized clients:")
        
        if show_details:
            # Collect detailed stats for each client
            client_details = []
            
            for client in clients:
                try:
                    # Get contribution details
                    details = blockchain_connector.get_client_contribution_details(client)
                    
                    # Add to list
                    client_details.append({
                        "address": client,
                        "contributions": details["contribution_count"],
                        "total_score": details["total_score"],
                        "rewards_earned": details["rewards_earned"],
                        "rewards_claimed": details["rewards_claimed"],
                        "last_contribution": datetime.fromtimestamp(details["last_contribution_timestamp"]).strftime("%Y-%m-%d %H:%M:%S") if details["last_contribution_timestamp"] > 0 else "Never"
                    })
                except Exception as e:
                    print(f"Error getting details for client {client}: {e}")
                    client_details.append({
                        "address": client,
                        "contributions": "Error",
                        "total_score": "Error",
                        "rewards_earned": "Error",
                        "rewards_claimed": "Error",
                        "last_contribution": "Error"
                    })
            
            # Print table
            headers = ["Address", "Contributions", "Total Score", "Rewards Earned", "Claimed", "Last Contribution"]
            table_data = [[d["address"], d["contributions"], d["total_score"], d["rewards_earned"], 
                           "Yes" if d["rewards_claimed"] else "No", d["last_contribution"]] for d in client_details]
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            # Simple list of addresses
            for i, client in enumerate(clients, 1):
                print(f"{i}. {client}")
    except Exception as e:
        print(f"Error listing clients: {e}")

def authorize_client(blockchain_connector: BlockchainConnector, client_address: str) -> None:
    """
    Authorize a client.
    
    Args:
        blockchain_connector: Blockchain connector instance
        client_address: Ethereum address of the client to authorize
    """
    try:
        # Check if client is already authorized
        if blockchain_connector.is_client_authorized(client_address):
            print(f"Client {client_address} is already authorized")
            return
        
        # Authorize client
        tx_hash = blockchain_connector.authorize_client(client_address)
        print(f"Client {client_address} authorized successfully")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error authorizing client: {e}")

def authorize_clients_from_file(blockchain_connector: BlockchainConnector, filepath: str) -> None:
    """
    Authorize clients from a file.
    
    Args:
        blockchain_connector: Blockchain connector instance
        filepath: Path to the file containing client addresses
    """
    try:
        # Read client addresses from file
        clients = []
        
        # Check file extension
        if filepath.endswith('.csv'):
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].startswith('0x'):
                        clients.append(row[0])
        else:
            # Assume text file with one address per line
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith('0x'):
                        clients.append(line)
        
        if not clients:
            print(f"No valid client addresses found in {filepath}")
            return
        
        print(f"Found {len(clients)} client addresses in {filepath}")
        
        # Filter out already authorized clients
        to_authorize = []
        for client in clients:
            if not blockchain_connector.is_client_authorized(client):
                to_authorize.append(client)
        
        if not to_authorize:
            print("All clients are already authorized")
            return
        
        print(f"Authorizing {len(to_authorize)} clients...")
        
        # Authorize clients in batches of 10
        batch_size = 10
        for i in range(0, len(to_authorize), batch_size):
            batch = to_authorize[i:i+batch_size]
            tx_hash = blockchain_connector.authorize_clients(batch)
            print(f"Authorized batch {i//batch_size + 1}/{(len(to_authorize)+batch_size-1)//batch_size}, tx: {tx_hash}")
        
        print(f"Successfully authorized {len(to_authorize)} clients")
    except Exception as e:
        print(f"Error authorizing clients from file: {e}")

def deauthorize_client(blockchain_connector: BlockchainConnector, client_address: str) -> None:
    """
    Deauthorize a client.
    
    Args:
        blockchain_connector: Blockchain connector instance
        client_address: Ethereum address of the client to deauthorize
    """
    try:
        # Check if client is authorized
        if not blockchain_connector.is_client_authorized(client_address):
            print(f"Client {client_address} is not authorized")
            return
        
        # Confirm deauthorization
        confirm = input(f"Are you sure you want to deauthorize client {client_address}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Deauthorization cancelled")
            return
        
        # Deauthorize client
        tx_hash = blockchain_connector.deauthorize_client(client_address)
        print(f"Client {client_address} deauthorized successfully")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error deauthorizing client: {e}")

def show_client_contributions(blockchain_connector: BlockchainConnector, client_address: str) -> None:
    """
    Show detailed contributions for a client.
    
    Args:
        blockchain_connector: Blockchain connector instance
        client_address: Ethereum address of the client
    """
    try:
        # Check if client is authorized
        if not blockchain_connector.is_client_authorized(client_address):
            print(f"Client {client_address} is not authorized")
            return
        
        # Get client contribution details
        details = blockchain_connector.get_client_contribution_details(client_address)
        
        print(f"Client {client_address} contribution summary:")
        print(f"  Total contributions: {details['contribution_count']}")
        print(f"  Total score: {details['total_score']}")
        print(f"  Rewards earned: {details['rewards_earned']}")
        print(f"  Rewards claimed: {'Yes' if details['rewards_claimed'] else 'No'}")
        
        if details["last_contribution_timestamp"] > 0:
            last_contrib = datetime.fromtimestamp(details["last_contribution_timestamp"])
            print(f"  Last contribution: {last_contrib.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("  Last contribution: Never")
        
        # Get contribution records
        records = blockchain_connector.get_client_contribution_records(client_address)
        
        if not records:
            print("No contribution records found")
            return
        
        print(f"\nContribution history ({len(records)} records):")
        
        # Prepare table data
        headers = ["Round", "Accuracy", "Score", "Timestamp", "Rewarded"]
        table_data = []
        
        for record in records:
            timestamp = datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            table_data.append([
                record["round"],
                f"{record['accuracy']:.2f}%",
                record["score"],
                timestamp,
                "Yes" if record["rewarded"] else "No"
            ])
        
        # Sort by round
        table_data.sort(key=lambda x: x[0])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except Exception as e:
        print(f"Error showing client contributions: {e}")

def show_round_contributions(blockchain_connector: BlockchainConnector, round_num: int) -> None:
    """
    Show all contributions for a specific round.
    
    Args:
        blockchain_connector: Blockchain connector instance
        round_num: Round number
    """
    try:
        # Get round contributions
        contributions = blockchain_connector.get_round_contributions(round_num)
        
        if not contributions:
            print(f"No contributions found for round {round_num}")
            return
        
        print(f"Found {len(contributions)} contributions for round {round_num}:")
        
        # Prepare table data
        headers = ["Client Address", "Accuracy", "Score", "Rewarded"]
        table_data = []
        
        for contrib in contributions:
            table_data.append([
                contrib["client_address"],
                f"{contrib['accuracy']:.2f}%",
                contrib["score"],
                "Yes" if contrib["rewarded"] else "No"
            ])
        
        # Sort by score (descending)
        table_data.sort(key=lambda x: x[2], reverse=True)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except Exception as e:
        print(f"Error showing round contributions: {e}")

def allocate_rewards(blockchain_connector: BlockchainConnector, round_num: int, total_reward: int) -> None:
    """
    Allocate rewards for a specific round.
    
    Args:
        blockchain_connector: Blockchain connector instance
        round_num: Round number
        total_reward: Total reward amount to distribute
    """
    try:
        # Check if there are contributions for this round
        contributions = blockchain_connector.get_round_contributions(round_num)
        
        if not contributions:
            print(f"No contributions found for round {round_num}")
            return
        
        # Check if rewards are already allocated
        if all(contrib["rewarded"] for contrib in contributions):
            print(f"Rewards for round {round_num} are already allocated")
            return
        
        # Confirm allocation
        print(f"About to allocate {total_reward} reward points among {len(contributions)} clients for round {round_num}")
        confirm = input("Are you sure you want to continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Reward allocation cancelled")
            return
        
        # Allocate rewards
        tx_hash = blockchain_connector.allocate_rewards_for_round(round_num, total_reward)
        print(f"Rewards allocated successfully for round {round_num}")
        print(f"Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Error allocating rewards: {e}")

def export_client_stats(blockchain_connector: BlockchainConnector, filepath: str) -> None:
    """
    Export client statistics to a file.
    
    Args:
        blockchain_connector: Blockchain connector instance
        filepath: Path to export the statistics
    """
    try:
        # Get all authorized clients
        clients = blockchain_connector.get_all_authorized_clients()
        
        if not clients:
            print("No authorized clients found")
            return
        
        # Collect stats for each client
        client_stats = {}
        
        for client in clients:
            try:
                # Get contribution details
                details = blockchain_connector.get_client_contribution_details(client)
                
                # Get contribution records
                records = blockchain_connector.get_client_contribution_records(client)
                
                # Store in stats
                client_stats[client] = {
                    "details": details,
                    "records": records
                }
            except Exception as e:
                print(f"Error getting stats for client {client}: {e}")
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(client_stats, f, indent=2)
            
        print(f"Exported client stats for {len(client_stats)} clients to {filepath}")
    except Exception as e:
        print(f"Error exporting client stats: {e}")

def main():
    parser = argparse.ArgumentParser(description="Client management tool for federated learning with blockchain")
    
    # Blockchain connection
    parser.add_argument("--ganache-url", type=str, default="http://192.168.1.146:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="EnhancedModelRegistry contract address")
    parser.add_argument("--private-key", type=str, help="Private key for blockchain transactions")
    
    # Actions
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # List clients
    list_parser = subparsers.add_parser("list", help="List all authorized clients")
    list_parser.add_argument("--details", action="store_true", help="Show detailed stats for each client")
    
    # Authorize client
    auth_parser = subparsers.add_parser("authorize", help="Authorize a client")
    auth_parser.add_argument("client_address", type=str, help="Ethereum address of the client to authorize")
    
    # Authorize clients from file
    auth_file_parser = subparsers.add_parser("authorize-file", help="Authorize clients from a file")
    auth_file_parser.add_argument("filepath", type=str, help="Path to the file containing client addresses")
    
    # Deauthorize client
    deauth_parser = subparsers.add_parser("deauthorize", help="Deauthorize a client")
    deauth_parser.add_argument("client_address", type=str, help="Ethereum address of the client to deauthorize")
    
    # Show client contributions
    client_contrib_parser = subparsers.add_parser("client-contributions", help="Show detailed contributions for a client")
    client_contrib_parser.add_argument("client_address", type=str, help="Ethereum address of the client")
    
    # Show round contributions
    round_contrib_parser = subparsers.add_parser("round-contributions", help="Show all contributions for a specific round")
    round_contrib_parser.add_argument("round_num", type=int, help="Round number")
    
    # Allocate rewards
    reward_parser = subparsers.add_parser("allocate-rewards", help="Allocate rewards for a specific round")
    reward_parser.add_argument("round_num", type=int, help="Round number")
    reward_parser.add_argument("--total-reward", type=int, default=1000, help="Total reward amount to distribute")
    
    # Export client stats
    export_parser = subparsers.add_parser("export", help="Export client statistics to a file")
    export_parser.add_argument("filepath", type=str, help="Path to export the statistics")
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        return
    
    # Load blockchain connector
    blockchain_connector = load_blockchain_connector(
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        private_key=args.private_key
    )
    
    # Perform the selected action
    if args.action == "list":
        list_clients(blockchain_connector, args.details)
    elif args.action == "authorize":
        authorize_client(blockchain_connector, args.client_address)
    elif args.action == "authorize-file":
        authorize_clients_from_file(blockchain_connector, args.filepath)
    elif args.action == "deauthorize":
        deauthorize_client(blockchain_connector, args.client_address)
    elif args.action == "client-contributions":
        show_client_contributions(blockchain_connector, args.client_address)
    elif args.action == "round-contributions":
        show_round_contributions(blockchain_connector, args.round_num)
    elif args.action == "allocate-rewards":
        allocate_rewards(blockchain_connector, args.round_num, args.total_reward)
    elif args.action == "export":
        export_client_stats(blockchain_connector, args.filepath)

if __name__ == "__main__":
    main()