"""
Configuration Settings for Federated Learning API
"""
import os
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Federated Learning API"
    DEBUG: bool = True
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Server module settings
    SERVER_MODULE_PATH: str = "../v2_fl/server.py"
    METRICS_DIR: str = "./metrics"
    
    V2_FL_PATH: str = "../v2_fl"
    V2_FL_SERVER_PATH: str = "../v2_fl/server.py"
    
    # Blockchain settings
    BLOCKCHAIN_URL: str = "http://192.168.1.146:7545"  # Ganache default
    CONTRACT_ADDRESS: str = "0x8e12169F51F3949a4b66eB09813F4b419e14e44e"  # Default contract address
    CONTRACT_ABI_PATH: str = "./app/assets/Federation.json"
    ADMIN_ADDRESS: str = "0xdE1a8A52252a7724aDfb1AcaC300Ea1b2c4eaFE0"  # Default admin address
    ADMIN_PRIVATE_KEY: str = "0xe6fefd5bdf1f22c50ad84dbde500cdf15b558c29c9b8c84f417cf7ed2b286934"  # Default admin key (DO NOT USE IN PRODUCTION)
    
    # IPFS settings
    IPFS_API_URL: str = "http://127.0.0.1:5001/api/v0"  # Default IPFS API 
    LOCAL_MODEL_DIR: str = "./models"
    
    @field_validator("SERVER_MODULE_PATH")
    @classmethod
    def validate_server_module_path(cls, v):
        """Validate that the server module path exists"""
        path = Path(v).resolve()
        if not path.exists():
            raise ValueError(f"Server module path {v} does not exist")
        return str(path)
    
    @field_validator("CONTRACT_ABI_PATH")
    @classmethod
    def validate_contract_abi_path(cls, v):
        """Validate that the contract ABI path exists or use default"""
        path = Path(v)
        if not path.exists():
            # Check if the file exists in the assets directory
            app_dir = Path(__file__).resolve().parent.parent
            alternate_path = app_dir / "assets" / "contract_abi.json"
            if alternate_path.exists():
                return str(alternate_path)
            else:
                # In development, we might not have the ABI yet
                print(f"Warning: Contract ABI file {v} does not exist")
                return v
        return str(path)
    
    @field_validator("METRICS_DIR", "LOCAL_MODEL_DIR")
    @classmethod
    def create_directory_if_not_exists(cls, v):
        """Create directory if it doesn't exist"""
        path = Path(v)
        os.makedirs(path, exist_ok=True)
        return str(path)
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


# Create global settings object
settings = Settings()