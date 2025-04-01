from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class ModelUpdate(BaseModel):
    model_weights: Dict
    metrics: str

class EvaluationMetrics(BaseModel):
    loss: int
    accuracy: int
    auc: int
    precision: int
    recall: int

class TrainingRound(BaseModel):
    round_id: int
    start_time: datetime
    end_time: Optional[datetime]
    global_model_hash: Optional[str]
    completed: bool
    participants: List[Dict]