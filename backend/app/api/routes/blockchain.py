from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.dependencies import get_current_user
from app.schemas.blockchain import Transaction, BlockchainStats
from app.services import blockchain_service

router = APIRouter(prefix="/blockchain", tags=["blockchain"])

@router.get("/transactions", response_model=List[Transaction])
async def get_transactions(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    return await blockchain_service.get_transactions(skip, limit)

@router.get("/stats", response_model=BlockchainStats)
async def get_blockchain_stats(
    current_user: User = Depends(get_current_user)
):
    return await blockchain_service.get_stats()

@router.get("/verify/{tx_hash}")
async def verify_transaction(
    tx_hash: str,
    current_user: User = Depends(get_current_user)
):
    result = await blockchain_service.verify_transaction(tx_hash)
    if not result:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result