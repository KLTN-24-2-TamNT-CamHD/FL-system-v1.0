from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.dependencies import get_current_user
from app.services import ipfs_service

router = APIRouter(prefix="/storage", tags=["storage"])

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    try:
        ipfs_hash = await ipfs_service.store_file(file)
        return {"ipfs_hash": ipfs_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retrieve/{ipfs_hash}")
async def retrieve_file(
    ipfs_hash: str,
    current_user: User = Depends(get_current_user)
):
    try:
        file_data = await ipfs_service.retrieve_file(ipfs_hash)
        return file_data
    except Exception as e:
        raise HTTPException(status_code=404, detail="File not found")