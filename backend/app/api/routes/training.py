from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.dependencies import verify_request
from app.services import blockchain_service
from app.services.fl_service import fl_service  # Updated import
from app.schemas.training import ModelUpdate, EvaluationMetrics
import subprocess
import os
from datetime import datetime

router = APIRouter()

# Keep your existing endpoints
@router.post("/rounds/initiate")
async def initiate_training_round(
    _: bool = Depends(verify_request)
):
    try:
        tx_hash = await blockchain_service.initiate_training_round()
        return {
            "message": "Training round initiated",
            "tx_hash": tx_hash,
            "timestamp": "2025-03-31 15:16:20",
            "deployer": "dinhcam89"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rounds/{round_id}/complete")
async def complete_round(
    round_id: int,
    global_model: ModelUpdate,
    _: bool = Depends(verify_request)
):
    try:
        ipfs_hash = global_model.model_weights
        tx_hash = await blockchain_service.complete_round(round_id, ipfs_hash)
        
        return {
            "message": "Round completed",
            "global_model_hash": ipfs_hash,
            "tx_hash": tx_hash,
            "timestamp": "2025-03-31 15:16:20",
            "deployer": "dinhcam89"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rounds/{round_id}")
async def get_round_info(
    round_id: int,
    _: bool = Depends(verify_request)
):
    try:
        round_info = await blockchain_service.get_training_round_info(round_id)
        return {
            **round_info,
            "timestamp": "2025-03-31 15:16:20",
            "deployer": "dinhcam89"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rounds/{round_id}/evaluate")
async def submit_evaluation(
    round_id: int,
    metrics: EvaluationMetrics,
    _: bool = Depends(verify_request)
):
    try:
        tx_hash = await blockchain_service.submit_evaluation(
            round_id,
            metrics.loss,
            metrics.accuracy,
            metrics.auc,
            metrics.precision,
            metrics.recall
        )
        
        return {
            "message": "Evaluation submitted",
            "tx_hash": tx_hash,
            "timestamp": "2025-03-31 15:16:20",
            "deployer": "dinhcam89"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fl/status")
async def get_fl_status(_: bool = Depends(verify_request)):
    """Get status of FL training"""
    try:
        status = fl_service.get_status()
        return status
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "blockchain_connected": blockchain_service.is_connected(),
            "timestamp": "2025-03-31 16:03:51",
            "admin": "dinhcam89"
        }

@router.post("/fl/start")
async def start_fl_training(_: bool = Depends(verify_request)):
    """Start Flower federated learning server"""
    try:
        # Check blockchain connection first
        if not blockchain_service.is_connected():
            return {
                "message": "Warning: Blockchain network not connected. Starting server anyway.",
                "blockchain_status": "disconnected",
                **fl_service.start_server(port=8080)
            }
        return fl_service.start_server(port=8080)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fl/stop")
async def stop_fl_training(_: bool = Depends(verify_request)):
    """Stop FL training server"""
    try:
        return fl_service.stop_server()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))