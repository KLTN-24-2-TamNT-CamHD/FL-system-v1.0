from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import logging
from app.api.routes import status, clients, training, blockchain
from app.core.fl_system import FLSystem
from app.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Load config
def load_config():
    config_path = os.getenv("CONFIG_PATH", "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        # Return default config
        return {
            "flower_server_address": "localhost:8080",
            "web3_provider_url": "http://localhost:8545",
            "ipfs_api_url": "http://localhost:5001",
            "dataset_path": "data/california_housing.csv",
            "api_port": 8000,
            "api_host": "0.0.0.0"
        }

config = load_config()

# Initialize FL system
fl_system = FLSystem(
    flower_server_address=config.get("flower_server_address", "localhost:8088"),
    web3_provider_url=config.get("web3_provider_url", "http://192.168.1.146:7545"),
    ipfs_api_url=config.get("ipfs_api_url", "http://localhost:5001/api/v0"),
    dataset_path=config.get("dataset_path", "data/california_housing.csv")
)

# Create FastAPI app
app = FastAPI(
    title="Federated Learning API",
    description="API server for managing Federated Learning with GA-Stacking, IPFS, and Blockchain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store FL system instance in app state
app.state.fl_system = fl_system

# Include routers
app.include_router(status.router, prefix="/api", tags=["Status"])
app.include_router(clients.router, prefix="/api", tags=["Clients"])
app.include_router(training.router, prefix="/api", tags=["Training"])
app.include_router(blockchain.router, prefix="/api", tags=["Blockchain"])

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing FL system...")
    await app.state.fl_system.initialize()
    logger.info("FL system initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FL system...")
    # Add any cleanup code here

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=config.get("api_host", "0.0.0.0"), 
        port=config.get("api_port", 8000),
        reload=True
    )