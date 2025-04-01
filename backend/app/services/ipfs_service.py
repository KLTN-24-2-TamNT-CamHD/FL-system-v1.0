import logging
from app.config import settings

logger = logging.getLogger(__name__)

class IPFSService:
    def __init__(self):
        self.client = None
        self.enabled = getattr(settings, 'IPFS_ENABLED', False)
        
        if self.enabled:
            try:
                import ipfshttpclient
                self.client = ipfshttpclient.connect(settings.IPFS_API_URL)
                logger.info("IPFS service initialized")
            except ImportError:
                logger.warning("ipfshttpclient not installed. IPFS functionality disabled.")
                self.enabled = False
            except Exception as e:
                logger.warning(f"IPFS service not available: {str(e)}")
                self.enabled = False

    async def store_model_weights(self, weights: dict) -> str:
        if not self.enabled:
            # Return mock hash for testing
            mock_hash = f"QmTest{settings.DEPLOYMENT_TIMESTAMP.replace(' ', '')}"
            logger.info(f"IPFS disabled. Returning mock hash: {mock_hash}")
            return mock_hash
            
        try:
            import json
            weights_json = json.dumps(weights)
            result = self.client.add_json(weights_json)
            return result
        except Exception as e:
            logger.error(f"Failed to store weights in IPFS: {str(e)}")
            raise

    async def retrieve_model_weights(self, ipfs_hash: str) -> dict:
        if not self.enabled:
            # Return mock weights for testing
            logger.info("IPFS disabled. Returning mock weights.")
            return {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}
            
        try:
            weights_json = self.client.get_json(ipfs_hash)
            return json.loads(weights_json)
        except Exception as e:
            logger.error(f"Failed to retrieve weights from IPFS: {str(e)}")
            raise

    def __del__(self):
        if hasattr(self, 'client') and self.client and getattr(self, 'enabled', False):
            try:
                self.client.close()
            except:
                pass

# Initialize the service
ipfs_service = IPFSService()