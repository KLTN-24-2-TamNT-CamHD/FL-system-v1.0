import logging
import asyncio
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

# Import our connector modules
from app.core.fl_connector import FlowerServerConnector
from app.core.blockchain_connector import BlockchainConnector

# Setup logging
logger = logging.getLogger(__name__)

class FLSystem:
    """
    Integration class that ties together all components of the Federated Learning system
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FLSystem()
        return cls._instance
    def __init__(self, 
                 flower_server_address: str = "localhost:8080",
                 web3_provider_url: str = "http://192.168.1.146:7545",
                 ipfs_api_url: str = "http://localhost:5001",
                 dataset_path: str = "data/california_housing.csv"):
        """
        Initialize the FL system with all necessary components
        """
        self.flower_connector = FlowerServerConnector(server_address=flower_server_address)
        self.blockchain_connector = BlockchainConnector()  # Placeholder for blockchain connector
        self.blockchain_initialized = False

        
        self.dataset_path = dataset_path
        self.current_training_id = None
        self.config = {
            "flower_server_address": flower_server_address,
            "web3_provider_url": web3_provider_url,
            "ipfs_api_url": ipfs_api_url,
            "dataset_path": dataset_path
        }
        
    async def initialize(self) -> bool:
        """
        Initialize the system components
        """
        try:
            # Initialize Flower connector
            flower_initialized = await self.flower_connector.initialize()
            if not flower_initialized:
                logger.error("Failed to initialize Flower connector")
                return False
            
            # Check if dataset exists
            if not os.path.exists(self.dataset_path):
                logger.warning(f"Dataset not found at {self.dataset_path}")
                # For initial implementation, we don't require the dataset to be present
            
            logger.info("FLSystem initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing FLSystem: {str(e)}")
            return False
    
    async def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new federated learning training session
        """
        try:
            # Prepare the configuration for the Flower server
            server_config = {
                # Map the API parameters to your Flower server parameters
                "num_rounds": config.get("num_rounds", 3),
                "min_fit_clients": config.get("min_fit_clients", 2),
                "min_evaluate_clients": config.get("min_evaluate_clients", 2),
                "fraction_fit": config.get("fraction_fit", 1.0),
                
                # Add specific parameters for your server if provided in the config
                "ipfs_url": config.get("ipfs_url", "http://127.0.0.1:5001/api/v0"),
                "ganache_url": config.get("ganache_url", "http://192.168.1.146:7545"),
                "contract_address": config.get("contract_address"),
                "private_key": config.get("private_key"),
                "deploy_contract": config.get("deploy_contract", False),
                "version_prefix": config.get("version_prefix", "1.0"),
                "authorized_clients_only": config.get("authorized_clients_only", False),
                "authorized_clients": config.get("authorized_clients"),
                "round_rewards": config.get("round_rewards", 1000),
                "device": config.get("device", "cpu")
            }
            
            # Start training on Flower server
            training_started = await self.flower_connector.start_server(server_config)
            
            if not training_started:
                return {"status": "error", "message": "Failed to start training"}
                
            # Generate a training ID
            self.current_training_id = f"training_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            return {
                "status": "success",
                "training_id": self.current_training_id,
                "message": "Training started successfully"
            }
            
        except Exception as e:
            logger.error(f"Error starting training: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def stop_training(self) -> Dict[str, Any]:
        """
        Stop the current training session
        """
        try:
            training_stopped = await self.flower_connector.stop_training()
            
            if not training_stopped:
                return {"status": "error", "message": "Failed to stop training"}
                
            return {
                "status": "success",
                "message": "Training stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Error stopping training: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the overall system status
        """
        try:
            # Get status from Flower server
            flower_status = await self.flower_connector.get_status()
            
            return {
                "status": "success",
                "flower_status": flower_status,
                "latest_training_id": self.current_training_id
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_training_history(self) -> Dict[str, Any]:
        """
        Get the training history
        """
        try:
            # Get training history from Flower connector
            history = await self.flower_connector.get_training_history()
            
            return {
                "status": "success",
                "history": history
            }
            
        except Exception as e:
            logger.error(f"Error getting training history: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def register_client(self, client_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new client with the system
        """
        try:
            registration_successful = await self.flower_connector.register_client(client_id, client_info)
            
            if not registration_successful:
                return {"status": "error", "message": "Failed to register client"}
                
            return {
                "status": "success",
                "client_id": client_id,
                "message": "Client registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Error registering client: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_server_logs(self, num_lines: int = 100) -> Dict[str, Any]:
        """
        Get logs from the Flower server
        """
        try:
            logs = await self.flower_connector.read_server_logs(num_lines)
            
            return {
                "status": "success",
                "logs": logs
            }
            
        except Exception as e:
            logger.error(f"Error getting server logs: {str(e)}")
            return {"status": "error", "message": str(e)}
        
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the training process
        This method is used by the /api/status endpoint
        """
        try:
            # Get status from Flower connector
            flower_status = await self.flower_connector.get_status()
            
            # Return the status in the expected format
            return {
                "status": "active" if flower_status["server_running"] else "unknown",
                "server_running": flower_status["server_running"],
                "current_round": flower_status["current_round"],
                "total_rounds": flower_status["total_rounds"],
                "started_at": flower_status["started_at"],
                "active_clients": flower_status["active_clients"],
                "training_id": self.current_training_id
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            # Return a fallback status
            return {
                "status": "unknown",
                "server_running": False,
                "current_round": 0,
                "total_rounds": 0,
                "started_at": None,
                "active_clients": 0,
                "training_id": self.current_training_id
            }
            
            


    # Blockchain related methods
    def init_blockchain(self, blockchain_url: str, contract_address: str = None, private_key: str = None) -> bool:
        """Initialize the blockchain connection with the given parameters."""
        try:
            # Update the blockchain connector with the provided URL
            self.blockchain_connector = BlockchainConnector(
                blockchain_url=blockchain_url,
                contract_address=contract_address,
                private_key=private_key
            )
            
            # Check if connected
            if not self.blockchain_connector.is_connected():
                logger.error(f"Failed to connect to blockchain at {blockchain_url}")
                return False
                
            logger.info(f"Successfully connected to blockchain at {blockchain_url}")
            
            # Initialize contract if address provided
            if contract_address:
                result = self.blockchain_connector.initialize_contract(contract_address)
                if result:
                    self.blockchain_initialized = True
                    logger.info(f"Contract initialized at {contract_address}")
                    return True
                else:
                    logger.error(f"Failed to initialize contract at {contract_address}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error initializing blockchain: {str(e)}")
            return False
    async def get_authorized_clients(self) -> Dict[str, Any]:
        """Get the list of authorized clients."""
        try:
            if not self.blockchain_initialized or not self.blockchain_connector.is_contract_initialized():
                return {"status": "error", "message": "Contract not initialized"}
                
            clients = self.blockchain_connector.get_authorized_clients()
            return {"status": "success", "clients": clients}
        except Exception as e:
            logger.error(f"Error getting authorized clients: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def authorize_client(self, client_address: str) -> Dict[str, Any]:
        """
        Authorize a client to participate in federated learning
        """
        try:
            return await self.blockchain_connector.authorize_client(client_address)
            
        except Exception as e:
            logger.error(f"Error authorizing client: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def revoke_client(self, client_address: str) -> Dict[str, Any]:
        """
        Revoke a client's authorization
        """
        try:
            return await self.blockchain_connector.revoke_client(client_address)
            
        except Exception as e:
            logger.error(f"Error revoking client: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_latest_model(self) -> Dict[str, Any]:
        """
        Get the latest model information from the blockchain
        """
        try:
            return await self.blockchain_connector.get_latest_model()
            
        except Exception as e:
            logger.error(f"Error getting latest model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_all_models(self) -> Dict[str, Any]:
        """
        Get all models information from the blockchain
        """
        try:
            return await self.blockchain_connector.get_all_models()
            
        except Exception as e:
            logger.error(f"Error getting all models: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_client_rewards(self, client_address: str) -> Dict[str, Any]:
        """
        Get the rewards earned by a specific client
        """
        try:
            return await self.blockchain_connector.get_client_rewards(client_address)
            
        except Exception as e:
            logger.error(f"Error getting client rewards: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def deploy_contract(self) -> Dict[str, Any]:
        """
        Deploy a new contract to the blockchain
        """
        try:
            return await self.blockchain_connector.deploy_contract()
            
        except Exception as e:
            logger.error(f"Error deploying contract: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def set_contract_address(self, contract_address: str) -> Dict[str, Any]:
        """
        Set or update the contract address
        """
        try:
            success = await self.blockchain_connector.set_contract_address(contract_address)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Contract address set to {contract_address}"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to set contract address"
                }
            
        except Exception as e:
            logger.error(f"Error setting contract address: {str(e)}")
            return {"status": "error", "message": str(e)}