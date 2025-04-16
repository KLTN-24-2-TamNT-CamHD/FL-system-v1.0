import requests
import json
import io
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

class IPFSService:
    """Service for interacting with IPFS for storing and retrieving model parameters."""
    
    def __init__(self, api_url: str):
        """
        Initialize the IPFS service with the API URL.
        
        Args:
            api_url: The URL of the IPFS API (e.g., 'http://localhost:5001/api/v0')
        """
        self.api_url = api_url
        self.logger = logging.getLogger("federated.ipfs")
    
    def add_file(self, content: bytes) -> str:
        """
        Add content to IPFS and return the content hash (CID).
        
        Args:
            content: The binary content to store on IPFS
            
        Returns:
            The IPFS content identifier (CID)
            
        Raises:
            Exception: If the IPFS operation fails
        """
        try:
            # Prepare the file-like object
            files = {
                'file': io.BytesIO(content)
            }
            
            # Upload to IPFS
            response = requests.post(
                f"{self.api_url}/add",
                files=files
            )
            
            if response.status_code != 200:
                self.logger.error(f"IPFS add failed with status {response.status_code}: {response.text}")
                raise Exception(f"IPFS add failed: {response.text}")
                
            result = response.json()
            cid = result.get('Hash')
            
            if not cid:
                raise Exception("No CID returned from IPFS")
                
            self.logger.info(f"Content added to IPFS with CID: {cid}")
            return cid
            
        except Exception as e:
            self.logger.error(f"Error adding content to IPFS: {str(e)}")
            raise
    
    def get_file(self, cid: str) -> bytes:
        """
        Retrieve content from IPFS by its content identifier (CID).
        
        Args:
            cid: The IPFS content identifier
            
        Returns:
            The binary content retrieved from IPFS
            
        Raises:
            Exception: If the IPFS operation fails
        """
        try:
            # NOTE: For IPFS retrieval, use GET method instead of POST
            # This is the key fix for the 405 error
            response = requests.get(
                f"{self.api_url}/cat?arg={cid}",
                # Don't use POST for cat endpoint
                # Don't use data={'arg': cid} with GET
            )
            
            if response.status_code != 200:
                self.logger.error(f"IPFS cat failed with status {response.status_code}: {response.text}")
                raise Exception(f"IPFS cat failed: {response.text}")
                
            self.logger.info(f"Content retrieved from IPFS with CID: {cid}")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error retrieving content from IPFS: {str(e)}")
            raise
    
    def pin_cid(self, cid: str) -> bool:
        """
        Pin a CID to ensure the content is retained on the IPFS node.
        
        Args:
            cid: The IPFS content identifier to pin
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.api_url}/pin/add",
                params={'arg': cid}
            )
            
            if response.status_code != 200:
                self.logger.error(f"IPFS pin failed with status {response.status_code}: {response.text}")
                return False
                
            self.logger.info(f"Content pinned on IPFS with CID: {cid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error pinning content on IPFS: {str(e)}")
            return False
    
    def store_parameters(self, parameters: Any) -> Optional[str]:
        """
        Store model parameters on IPFS.
        
        Args:
            parameters: The model parameters to store (numpy arrays, lists, etc.)
            
        Returns:
            The IPFS content identifier (CID) or None if failed
        """
        try:
            # Convert parameters to bytes
            parameters_bytes = self._serialize_parameters(parameters)
            
            # Store on IPFS
            cid = self.add_file(parameters_bytes)
            
            # Ensure content is pinned
            self.pin_cid(cid)
            
            return cid
            
        except Exception as e:
            self.logger.error(f"Error storing parameters on IPFS: {str(e)}")
            return None
    
    def retrieve_parameters(self, cid: str) -> Optional[Any]:
        """
        Retrieve model parameters from IPFS.
        
        Args:
            cid: The IPFS content identifier
            
        Returns:
            The model parameters or None if failed
        """
        try:
            # Get content from IPFS
            content = self.get_file(cid)
            
            # Deserialize parameters
            parameters = self._deserialize_parameters(content)
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error retrieving parameters from IPFS: {str(e)}")
            return None
    
    def _serialize_parameters(self, parameters: Any) -> bytes:
        """
        Serialize model parameters to bytes.
        
        Args:
            parameters: The model parameters to serialize
            
        Returns:
            The serialized parameters as bytes
        """
        try:
            # For numpy arrays or list of numpy arrays
            if isinstance(parameters, list) and all(isinstance(p, np.ndarray) for p in parameters):
                # Save arrays to an in-memory buffer
                buffer = io.BytesIO()
                np.savez(buffer, *parameters)
                buffer.seek(0)
                return buffer.read()
            else:
                # For other types, use JSON
                return json.dumps(parameters).encode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Error serializing parameters: {str(e)}")
            raise
    
    def _deserialize_parameters(self, data: bytes) -> Any:
        """
        Deserialize bytes back to model parameters.
        
        Args:
            data: The serialized parameters bytes
            
        Returns:
            The deserialized model parameters
        """
        try:
            # Try to deserialize as numpy arrays first
            buffer = io.BytesIO(data)
            try:
                npz = np.load(buffer, allow_pickle=True)
                # Extract arrays from npz
                return [npz[key] for key in npz.files]
            except:
                # If not numpy, try JSON
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            self.logger.error(f"Error deserializing parameters: {str(e)}")
            raise