from fastapi import APIRouter, Request, HTTPException, Path, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.api.models.client import (
    ClientBase,
    ClientCreate,
    Client,
    ClientResponse
)

router = APIRouter()

# In-memory client storage (replace with database in production)
clients_db = {}

@router.get("/clients", response_model=List[Client])
async def list_clients():
    """
    Get list of all registered clients
    """
    return list(clients_db.values())

@router.get("/clients/{client_id}", response_model=Client)
async def get_client(client_id: str = Path(..., description="The ID of the client to get")):
    """
    Get information about a specific client
    """
    if client_id not in clients_db:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return clients_db[client_id]

@router.post("/clients/register", response_model=ClientResponse)
async def register_client(
    client: ClientCreate,
    request: Request
):
    """
    Register a new client with the system
    """
    fl_system = request.app.state.fl_system
    
    # Generate client ID
    client_id = f"client_{len(clients_db) + 1}"
    
    # Create client record
    new_client = Client(
        id=client_id,
        name=client.name,
        dataset_split=client.dataset_split,
        overlap_percentage=client.overlap_percentage,
        last_active=datetime.now()
    )
    
    # Register with FL system
    result = await fl_system.register_client(client_id, client.dict())
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to register client"))
    
    # Store client in database
    clients_db[client_id] = new_client
    
    return ClientResponse(
        status="success",
        message="Client registered successfully",
        client_id=client_id,
        client=new_client
    )

@router.put("/clients/{client_id}", response_model=ClientResponse)
async def update_client(
    client_update: ClientBase,
    client_id: str = Path(..., description="The ID of the client to update")
):
    """
    Update client configuration
    """
    if client_id not in clients_db:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Update client record
    client = clients_db[client_id]
    client.name = client_update.name
    client.dataset_split = client_update.dataset_split
    client.overlap_percentage = client_update.overlap_percentage
    client.last_active = datetime.now()
    
    return ClientResponse(
        status="success",
        message="Client updated successfully",
        client_id=client_id,
        client=client
    )

@router.delete("/clients/{client_id}", response_model=ClientResponse)
async def delete_client(client_id: str = Path(..., description="The ID of the client to delete")):
    """
    Remove a client from the system
    """
    if client_id not in clients_db:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Remove from database
    del clients_db[client_id]
    
    return ClientResponse(
        status="success",
        message=f"Client {client_id} deleted successfully"
    )