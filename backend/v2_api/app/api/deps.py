"""
Dependency injection for FastAPI routes
"""
from app.core.fl_connector import FlowerServerConnector
from app.services.fl_service import FederatedLearningService
from app.services.blockchain_service import BlockchainService
from app.services.ipfs_service import IPFSService

# Create singleton instances
_fl_connector = None
_blockchain_service = None
_ipfs_service = None
_fl_service = None

def get_fl_connector():
    """Get singleton FlowerServerConnector instance"""
    global _fl_connector
    if _fl_connector is None:
        _fl_connector = FlowerServerConnector()
    return _fl_connector

def get_blockchain_service():
    """Get singleton BlockchainService instance"""
    global _blockchain_service
    if _blockchain_service is None:
        _blockchain_service = BlockchainService()
    return _blockchain_service

def get_ipfs_service():
    """Get singleton IPFSService instance"""
    global _ipfs_service
    if _ipfs_service is None:
        _ipfs_service = IPFSService()
    return _ipfs_service

def get_fl_service():
    """Get singleton FederatedLearningService instance"""
    global _fl_service
    if _fl_service is None:
        _fl_service = FederatedLearningService(
            fl_connector=get_fl_connector(),
            blockchain_service=get_blockchain_service(),
            ipfs_service=get_ipfs_service()
        )
    return _fl_service