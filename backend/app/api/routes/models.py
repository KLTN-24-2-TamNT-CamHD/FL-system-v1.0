from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.dependencies import get_current_user
from app.schemas.models import ModelMetadata, ModelVersion
from app.services import ipfs_service, blockchain_service

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/versions", response_model=List[ModelVersion])
async def list_model_versions(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    return await blockchain_service.get_model_versions(skip, limit)

@router.get("/{version_id}", response_model=ModelMetadata)
async def get_model_version(
    version_id: int,
    current_user: User = Depends(get_current_user)
):
    model_data = await blockchain_service.get_model_version(version_id)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model version not found")
    return model_data

@router.post("/upload-initial", response_model=ModelMetadata)
async def upload_initial_model(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can upload initial models"
        )
    
    # Store model in IPFS
    ipfs_hash = await ipfs_service.store_file(file)
    
    # Record in blockchain
    tx_hash = await blockchain_service.record_initial_model(ipfs_hash)
    
    return {
        "version": 0,
        "ipfs_hash": ipfs_hash,
        "tx_hash": tx_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "uploaded_by": current_user.id
    }