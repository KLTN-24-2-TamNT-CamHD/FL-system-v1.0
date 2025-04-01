from pydantic_settings import BaseSettings
from typing import List
from datetime import datetime

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Federated Learning System"
    
    # Blockchain Settings
    WEB3_PROVIDER_URI: str = "http://127.0.0.1:7545"
    CONTRACT_ADDRESS: str = ""
    ADMIN_PRIVATE_KEY: str = ""
    
    # IPFS Settings (disabled by default)
    IPFS_ENABLED: bool = False
    IPFS_API_URL: str = "http://localhost:5001"
    
    # Deployment Info
    DEPLOYMENT_TIMESTAMP: str = "2025-03-31 14:09:48"
    DEPLOYER: str = "dinhcam89"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        env_file = ".env"

settings = Settings()