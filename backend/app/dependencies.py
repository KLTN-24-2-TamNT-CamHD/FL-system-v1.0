from fastapi import HTTPException
from web3 import Web3

def verify_ethereum_address(address: str) -> bool:
    """Simple validation for Ethereum addresses"""
    return Web3.is_address(address)

# We'll use this instead of admin token verification
async def verify_request():
    return True