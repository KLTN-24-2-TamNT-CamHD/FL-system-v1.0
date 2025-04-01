from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.dependencies import get_current_user
from app.schemas.analytics import (
    TrainingMetrics,
    InstitutionMetrics,
    SystemMetrics
)
from app.services import analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/training-metrics", response_model=List[TrainingMetrics])
async def get_training_metrics(
    start_date: str = None,
    end_date: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await analytics_service.get_training_metrics(
        db,
        start_date,
        end_date,
        current_user.institution_id
    )

@router.get("/institution-metrics", response_model=List[InstitutionMetrics])
async def get_institution_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can view institution metrics"
        )
    return await analytics_service.get_institution_metrics(db)

@router.get("/system-metrics", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can view system metrics"
        )
    return await analytics_service.get_system_metrics(db)