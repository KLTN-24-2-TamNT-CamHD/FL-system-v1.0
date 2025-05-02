# Updated blockchain_endpoints.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from app.core.fl_system import FLSystem

router = APIRouter(tags=["Blockchain"])

# Models for request data
class BlockchainInitRequest(BaseModel):
    blockchain_url: str = Field(default="http://192.168.1.146:7545", description="URL of the blockchain node")
    contract_address: Optional[str] = Field(default=None, description="Ethereum contract address")
    private_key: Optional[str] = Field(default=None, description="Private key for transactions")

class DeployContractRequest(BaseModel):
    private_key: str = Field(..., description="Private key for contract deployment")
    blockchain_url: Optional[str] = Field(default=None, description="Optional URL of the blockchain node")

class AuthorizeClientRequest(BaseModel):
    client_address: str = Field(..., description="Ethereum address of the client to authorize")
    account_address: Optional[str] = Field(default=None, description="Address to send the transaction from")

class RevokeClientRequest(BaseModel):
    client_address: str = Field(..., description="Ethereum address of the client to revoke")
    account_address: Optional[str] = Field(default=None, description="Address to send the transaction from")

# Get system instance
def get_system():
    return FLSystem.get_instance()

@router.post("/init-blockchain")
async def init_blockchain(request: BlockchainInitRequest, system: FLSystem = Depends(get_system)):
    """Initialize the blockchain connection with the given parameters."""
    result = system.init_blockchain(
        blockchain_url=request.blockchain_url,
        contract_address=request.contract_address,
        private_key=request.private_key
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to initialize blockchain connection")
    
    return {"status": "success", "message": "Blockchain connection initialized", 
            "contract_initialized": system.blockchain_initialized}

@router.post("/deploy-contract")
async def deploy_contract(request: DeployContractRequest, system: FLSystem = Depends(get_system)):
    """Deploy a new contract and return its address."""
    # Update blockchain URL if provided
    if request.blockchain_url:
        system.init_blockchain(blockchain_url=request.blockchain_url)
    
    contract_address = system.deploy_contract(request.private_key)
    
    if not contract_address:
        raise HTTPException(status_code=500, detail="Failed to deploy contract")
    
    return {"status": "success", "contract_address": contract_address}

@router.get("/authorized-clients")
async def get_authorized_clients(system: FLSystem = Depends(get_system)):
    """Get the list of authorized clients."""
    result = await system.get_authorized_clients()
    
    if result["status"] == "error":
        return result
    
    return {"status": "success", "clients": result["clients"]}

@router.post("/authorize-client")
async def authorize_client(request: AuthorizeClientRequest, system: FLSystem = Depends(get_system)):
    """Authorize a client to participate in federated learning."""
    result = system.authorize_client(
        client_address=request.client_address,
        account_address=request.account_address
    )
    
    return result

@router.post("/revoke-client")
async def revoke_client(request: RevokeClientRequest, system: FLSystem = Depends(get_system)):
    """Revoke a client's authorization."""
    result = system.revoke_client(
        client_address=request.client_address,
        account_address=request.account_address
    )
    
    return result