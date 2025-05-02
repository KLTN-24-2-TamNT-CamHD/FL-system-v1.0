"""
Test script to verify connections to IPFS and Blockchain (Ganache) from the API server.
Run this script directly to test the connections.
"""

import sys
import os
import time
from fastapi.testclient import TestClient
from app.main import app
from app.services.ipfs_service import IPFSService
from app.services.blockchain_service import BlockchainService
from app.core.config import settings

client = TestClient(app)

def test_ipfs_connection():
    """Test the connection to IPFS."""
    print("\n===== Testing IPFS Connection =====")
    try:
        ipfs_service = IPFSService(settings)
        
        # Try to add a simple file to IPFS to test connection
        test_content = b"This is a test file for IPFS connection"
        print("Adding test file to IPFS...")
        cid = ipfs_service.add_file(test_content)
        print(f"✅ Successfully connected to IPFS. Added test file with CID: {cid}")
        
        # Try to retrieve the file to confirm bidirectional communication
        print("Retrieving test file from IPFS...")
        retrieved_content = ipfs_service.get_file(cid)
        if retrieved_content == test_content:
            print("✅ Successfully retrieved test file from IPFS.")
        else:
            print("❌ Retrieved file content doesn't match original content.")
            
        return True
    except Exception as e:
        print(f"❌ Failed to connect to IPFS: {str(e)}")
        return False

def test_blockchain_connection():
    """Test the connection to Blockchain (Ganache)."""
    print("\n===== Testing Blockchain Connection =====")
    try:
        blockchain_service = BlockchainService(settings)
        
        # Get accounts to check the connection
        print("Fetching accounts from blockchain...")
        accounts = blockchain_service.w3.eth.accounts
        if accounts:
            print(f"✅ Successfully connected to Blockchain. Found {len(accounts)} accounts.")
            print(f"First account: {accounts[0]}")
            
            # Check balance of the first account
            balance = blockchain_service.w3.eth.get_balance(accounts[0])
            print(f"Balance of first account: {blockchain_service.w3.from_wei(balance, 'ether')} ETH")
            
            # Check if contract is accessible
            try:
                contract_address = blockchain_service.model_registry_contract.address
                print(f"✅ Successfully connected to the ModelRegistry contract at: {contract_address}")
                
                # Optional: Test a read operation on the contract
                try:
                    owner = blockchain_service.model_registry_contract.functions.owner().call()
                    print(f"✅ Contract owner address: {owner}")
                except Exception as read_err:
                    print(f"❌ Failed to read from contract: {str(read_err)}")
                    
            except Exception as contract_err:
                print(f"❌ Failed to connect to the ModelRegistry contract: {str(contract_err)}")
                
            return True
        else:
            print("❌ No accounts found on the blockchain.")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Blockchain: {str(e)}")
        return False

def test_api_endpoints():
    """Test the API endpoints for IPFS and Blockchain."""
    print("\n===== Testing API Endpoints =====")
    
    # Test IPFS endpoint
    print("Testing IPFS endpoints via API...")
    try:
        # This assumes you have an endpoint to check IPFS status
        response = client.get("/api/v1/ipfs/status")
        if response.status_code == 200:
            print(f"✅ IPFS status endpoint working: {response.json()}")
        else:
            print(f"❌ IPFS status endpoint failed with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing IPFS endpoint: {str(e)}")
    
    # Test Blockchain endpoint
    print("\nTesting Blockchain endpoints via API...")
    try:
        # This assumes you have an endpoint to check blockchain status
        response = client.get("/api/v1/blockchain/status")
        if response.status_code == 200:
            print(f"✅ Blockchain status endpoint working: {response.json()}")
        else:
            print(f"❌ Blockchain status endpoint failed with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing Blockchain endpoint: {str(e)}")

if __name__ == "__main__":
    print("====================================")
    print("Testing API Server Connections")
    print("====================================")
    
    ipfs_success = test_ipfs_connection()
    blockchain_success = test_blockchain_connection()
    
    # Optionally test the API endpoints if both connections are successful
    if ipfs_success and blockchain_success:
        print("\n✅ Both IPFS and Blockchain connections are working!")
        print("Now testing API endpoints...")
        time.sleep(1)  # Small delay for better readability
        test_api_endpoints()
    else:
        print("\n❌ One or more connections failed. Please fix the issues before testing API endpoints.")
    
    print("\n====================================")
    print("Connection Test Summary:")
    print(f"IPFS Connection: {'✅ Success' if ipfs_success else '❌ Failed'}")
    print(f"Blockchain Connection: {'✅ Success' if blockchain_success else '❌ Failed'}")
    print("====================================")
