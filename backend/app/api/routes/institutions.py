from fastapi import APIRouter, Depends, HTTPException
from app.schemas.institution import InstitutionCreate
from app.services import blockchain_service
from app.dependencies import verify_request, verify_ethereum_address

router = APIRouter()

@router.post("/register")
async def register_institution(
    institution: InstitutionCreate,
    _: bool = Depends(verify_request)  # Simplified verification
):
    if not verify_ethereum_address(institution.address):
        raise HTTPException(status_code=400, detail="Invalid Ethereum address")
        
    try:
        tx_hash = await blockchain_service.register_institution(
            institution.address,
            institution.name
        )
        return {
            "message": "Institution registered",
            "tx_hash": tx_hash,
            "timestamp": "2025-03-31 14:00:44",
            "deployer": "dinhcam89"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))