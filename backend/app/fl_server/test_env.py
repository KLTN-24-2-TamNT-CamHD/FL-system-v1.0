# test_services.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.blockchain_service import BlockchainService
from services.ipfs_service import IPFSService
import json
import torch
import numpy as np

# Load config
with open('client_config.json', 'r') as f:
    config = json.load(f)

# Test IPFS
print("Testing IPFS connection...")
ipfs = IPFSService(config["ipfs"]["api_url"])
test_data = {"params": [np.array([1.0, 2.0, 3.0])], "metadata": {"test": True}}
try:
    ipfs_hash = ipfs.store_model_params(test_data["params"], test_data["metadata"])
    print(f"Successfully stored test data on IPFS: {ipfs_hash}")
    
    # Try retrieving
    retrieved = ipfs.retrieve_model_params(ipfs_hash)
    print(f"Successfully retrieved data from IPFS")
except Exception as e:
    print(f"IPFS test failed: {e}")

# Test Blockchain
print("\nTesting Blockchain connection...")
bc = BlockchainService(
    provider_url=config["blockchain"]["provider_url"],
    contract_address=config["blockchain"]["contract_address"],
    contract_abi_path=config["blockchain"]["contract_abi_path"],
    private_key=config["blockchain"]["private_key"],
    account_address=config["blockchain"]["account_address"]
)
try:
    # Test a read operation
    current_round = bc.get_current_round()
    print(f"Current round from blockchain: {current_round}")
    
    # Test is_authorized
    is_auth = bc.is_institution_authorized(config["blockchain"]["account_address"])
    print(f"Is institution authorized: {is_auth}")
except Exception as e:
    print(f"Blockchain test failed: {e}")
