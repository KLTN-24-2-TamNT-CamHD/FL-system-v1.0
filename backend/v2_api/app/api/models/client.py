from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
from datetime import datetime

class ClientBase(BaseModel):
    """Base model for client data"""
    name: str
    dataset_split: str = "overlapping"
    overlap_percentage: Optional[float] = 30.0
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Client-1",
                    "dataset_split": "overlapping",
                    "overlap_percentage": 30.0
                }
            ]
        }
    }

class ClientCreate(ClientBase):
    """Model for client creation"""
    pass

class Client(ClientBase):
    """Model for client information"""
    id: str
    status: str = "idle"
    last_active: Optional[datetime] = None
    
    model_config = {
        "from_attributes": True
    }

class ClientResponse(BaseModel):
    """Response for client operations"""
    status: str
    message: Optional[str] = None
    client_id: Optional[str] = None
    client: Optional[Client] = None