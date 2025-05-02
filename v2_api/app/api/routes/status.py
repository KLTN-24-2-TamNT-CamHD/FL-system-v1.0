from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/status")
async def get_status(request: Request) -> Dict[str, Any]:
    """
    Get current system status including Flower server, blockchain, and IPFS components
    """
    fl_system = request.app.state.fl_system
    
    # Get system status
    result = await fl_system.get_system_status()
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to get system status"))
    
    # Transform the response for the API
    flower_status = result.get("flower_status", {})
    
    return {
        "status": flower_status.get("status", "unknown"),
        "server_running": flower_status.get("server_running", False),
        "current_round": flower_status.get("current_round", 0),
        "total_rounds": flower_status.get("total_rounds", 0),
        "started_at": flower_status.get("started_at"),
        "active_clients": flower_status.get("active_clients", 0),
        "training_id": result.get("latest_training_id")
    }

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint to verify API is running
    """
    return {"status": "healthy"}

@router.get("/info")
async def system_info(request: Request) -> Dict[str, Any]:
    """
    Get information about the system configuration
    """
    fl_system = request.app.state.fl_system
    
    # Return system configuration information
    return {
        "flower_server_address": fl_system.config.get("flower_server_address"),
        "blockchain_enabled": bool(fl_system.config.get("web3_provider_url")),
        "ipfs_enabled": bool(fl_system.config.get("ipfs_api_url")),
        "dataset_path": fl_system.config.get("dataset_path"),
        "api_version": "1.0.0"
    }