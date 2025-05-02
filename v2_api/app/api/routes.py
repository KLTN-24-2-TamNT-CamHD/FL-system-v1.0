"""
API route registration
"""
from fastapi import APIRouter
from app.api.endpoints import training_endpoints, blockchain_endpoints, ipfs_endpoints

# Create main API router
api_router = APIRouter()

# Include all endpoint routers with appropriate prefixes and tags
api_router.include_router(
    training_endpoints.router, 
    prefix="/training", 
    tags=["training"]
)

api_router.include_router(
    blockchain_endpoints.router, 
    prefix="/blockchain", 
    tags=["blockchain"]
)

api_router.include_router(
    ipfs_endpoints.router, 
    prefix="/ipfs", 
    tags=["ipfs"]
)