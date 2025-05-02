from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Request
from typing import List, Dict, Any, Optional

from app.api.models.training import (
    TrainingConfig, 
    TrainingStatus, 
    TrainingResponse, 
    TrainingHistory
)

router = APIRouter()

@router.post("/start-training", response_model=TrainingResponse)
async def start_training(
    request: Request,
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """
    Start a new federated learning training session
    """
    fl_system = request.app.state.fl_system
    
    # Start the training
    result = await fl_system.start_training(config.dict())
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return result

@router.post("/stop-training", response_model=TrainingResponse)
async def stop_training(request: Request):
    """
    Stop the current training session
    """
    fl_system = request.app.state.fl_system
    
    # Stop the training
    result = await fl_system.stop_training()
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return result

@router.get("/training-status", response_model=TrainingStatus)
async def get_training_status(request: Request):
    """
    Get the current status of the training process
    """
    fl_system = request.app.state.fl_system
    
    # Get system status
    result = await fl_system.get_system_status()
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to get system status"))
        
    # Extract the flower status and return it
    flower_status = result.get("flower_status", {})
    
    return TrainingStatus(
        status=flower_status.get("status", "unknown"),
        server_running=flower_status.get("server_running", False),
        current_round=flower_status.get("current_round", 0),
        total_rounds=flower_status.get("total_rounds", 0),
        started_at=flower_status.get("started_at"),
        active_clients=flower_status.get("active_clients", 0),
        client_statuses=flower_status.get("client_statuses", {})
    )

@router.get("/training-history", response_model=TrainingHistory)
async def get_training_history(request: Request):
    """
    Get the history of training rounds
    """
    fl_system = request.app.state.fl_system
    
    # Get training history
    result = await fl_system.get_training_history()
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to get training history"))
        
    return result

@router.get("/server-logs")
async def get_server_logs(
    request: Request,
    num_lines: int = Query(100, description="Number of log lines to retrieve")
):
    """
    Get the Flower server logs
    """
    fl_system = request.app.state.fl_system
    
    # Get server logs
    result = await fl_system.get_server_logs(num_lines)
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to get server logs"))
        
    return result