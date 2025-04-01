from pydantic import BaseModel
from typing import Optional

class InstitutionCreate(BaseModel):
    address: str
    name: str