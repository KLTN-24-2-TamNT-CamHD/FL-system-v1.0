import json
import os
from typing import Dict, List, Any, Optional, Tuple
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
import logging

logger = logging.getLogger("federated.blockchain")

class BlockchainService:
    """Service to interact with the Ethereum blockchain and smart contract for federated learning."""
    
    def __init__(
        self, 
        provider_url: str, 
        contract_address: str, 
        contract_abi_path: str, 
        private_key: Optional[str] = None,
        account_address: Optional[str] = None
    ):
        """
        Initialize blockchain service with connection to Ethereum and contract.
        
        Args:
            provider_url: URL of the Ethereum node (http, ws, or ipc)
            contract_address: Address of the deployed FraudDetectionFederated contract
            contract_abi_path: Path to the contract ABI JSON file
            private_key: Private key for transactions (optional for read-only)
            account_address: Ethereum address (derived from private key if not provided)
        """
        self.provider_url = provider_url
        self.contract_address = Web3.to_checksum_address(contract_address)
        
        # Connect to Ethereum node
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        
        # Add middleware for compatibility with PoA chains like Goerli
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Check connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum node at {provider_url}")
        
        # Setup account for transactions
        self.private_key = private_key
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            self.account_address = self.account.address
        else:
            self.account = None
            self.account_address = account_address if account_address else None
            
        logger.info(f"Connected to blockchain at {provider_url}")
        logger.info(f"Using account: {self.account_address}")
        
        # Load contract ABI and create contract instance
        with open(contract_abi_path, 'r') as f:
            contract_abi = json.load(f)
        
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=contract_abi
        )
        
        logger.info(f"Contract instance created at {contract_address}")
    
    def _get_transaction_params(self, gas_limit: int = 2000000) -> Dict[str, Any]:
        """Get transaction parameters including nonce management."""
        return {
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': gas_limit,
            'gasPrice': self.w3.eth.gas_price
        }
    
    def _send_transaction(self, signed_tx):
        """
        Send a signed transaction with compatibility for different Web3.py versions.
        
        Args:
            signed_tx: The signed transaction object
            
        Returns:
            Transaction hash
        """
        try:
            # For newer Web3.py versions
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        except AttributeError:
            try:
                # For older Web3.py versions
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            except Exception as e:
                # Fallback for other potential issues
                raise ValueError(f"Failed to send transaction: {e}")
        
        return tx_hash
    
    def register_institution(self, institution_address: str, name: str) -> str:
        """
        Register an institution in the FL system (onlyOwner function).
        
        Args:
            institution_address: Ethereum address of the institution
            name: Name of the institution
            
        Returns:
            Transaction hash
        """
        if not self.private_key:
            raise ValueError("Private key required for transaction")
        
        institution_address = Web3.to_checksum_address(institution_address)
        
        tx_params = self._get_transaction_params()
        
        # Build transaction
        tx = self.contract.functions.registerInstitution(
            institution_address, 
            name
        ).build_transaction(tx_params)
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        
        # Send transaction
        tx_hash = self._send_transaction(signed_tx)
        
        # Wait for transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status != 1:
            raise ValueError(f"Transaction failed: {tx_receipt}")
        
        logger.info(f"Institution registered: {institution_address} ({name})")
        return tx_hash.hex()
    
    def initiate_training_round(self) -> Tuple[str, int]:
        """
        Start a new training round (onlyOwner function).
        
        Returns:
            Tuple of (transaction hash, round ID)
        """
        if not self.private_key:
            raise ValueError("Private key required for transaction")
        
        # Get current round before transaction
        current_round = self.get_current_round()
        
        tx_params = self._get_transaction_params()
        
        # Build transaction
        tx = self.contract.functions.initiateTrainingRound().build_transaction(tx_params)
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        
        # Send transaction
        tx_hash = self._send_transaction(signed_tx)
        
        # Wait for transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status != 1:
            raise ValueError(f"Transaction failed: {tx_receipt}")
        
        # New round should be current_round + 1
        new_round = current_round + 1
        logger.info(f"Training round {new_round} initiated")
        
        return (tx_hash.hex(), new_round)
    
    def submit_model_update(self, round_id: int, ipfs_hash: str, metrics: str) -> str:
        """
        Submit a model update for a training round (onlyAuthorized function).
        
        Args:
            round_id: ID of the current training round
            ipfs_hash: IPFS hash of the model parameters
            metrics: JSON string with metrics
            
        Returns:
            Transaction hash
        """
        if not self.private_key:
            raise ValueError("Private key required for transaction")
        
        tx_params = self._get_transaction_params()
        
        # Build transaction
        tx = self.contract.functions.submitModelUpdate(
            round_id,
            ipfs_hash,
            metrics
        ).build_transaction(tx_params)
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        
        # Send transaction
        tx_hash = self._send_transaction(signed_tx)
        
        # Wait for transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status != 1:
            raise ValueError(f"Transaction failed: {tx_receipt}")
        
        logger.info(f"Model update submitted for round {round_id}: {ipfs_hash}")
        return tx_hash.hex()
    
    def complete_round(self, round_id: int, global_model_hash: str) -> str:
        """
        Complete a training round with the global model (onlyOwner function).
        
        Args:
            round_id: ID of the training round to complete
            global_model_hash: IPFS hash of the global model
            
        Returns:
            Transaction hash
        """
        if not self.private_key:
            raise ValueError("Private key required for transaction")
        
        tx_params = self._get_transaction_params()
        
        # Build transaction
        tx = self.contract.functions.completeRound(
            round_id,
            global_model_hash
        ).build_transaction(tx_params)
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        
        # Send transaction
        tx_hash = self._send_transaction(signed_tx)
        
        # Wait for transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status != 1:
            raise ValueError(f"Transaction failed: {tx_receipt}")
        
        logger.info(f"Training round {round_id} completed with global model: {global_model_hash}")
        return tx_hash.hex()
    
    def submit_evaluation(
        self, 
        round_id: int, 
        loss: float, 
        accuracy: float, 
        auc: float, 
        precision: float, 
        recall: float
    ) -> str:
        """
        Submit model evaluation metrics (onlyAuthorized function).
        
        Args:
            round_id: ID of the training round
            loss: Loss value (converted to uint256)
            accuracy: Accuracy value (converted to uint256)
            auc: AUC value (converted to uint256)
            precision: Precision value (converted to uint256)
            recall: Recall value (converted to uint256)
            
        Returns:
            Transaction hash
        """
        if not self.private_key:
            raise ValueError("Private key required for transaction")
        
        # Convert float metrics to uint256 (multiply by 1000 to preserve 3 decimals)
        loss_uint = int(loss * 1000)
        accuracy_uint = int(accuracy * 1000)
        auc_uint = int(auc * 1000)
        precision_uint = int(precision * 1000)
        recall_uint = int(recall * 1000)
        
        tx_params = self._get_transaction_params()
        
        # Build transaction
        tx = self.contract.functions.submitEvaluation(
            round_id,
            loss_uint,
            accuracy_uint,
            auc_uint,
            precision_uint,
            recall_uint
        ).build_transaction(tx_params)
        
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        
        # Send transaction
        tx_hash = self._send_transaction(signed_tx)
        
        # Wait for transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status != 1:
            raise ValueError(f"Transaction failed: {tx_receipt}")
        
        logger.info(f"Evaluation submitted for round {round_id}")
        return tx_hash.hex()
    
    # Read-only methods
    
    def is_institution_authorized(self, address: str) -> bool:
        """Check if an institution is authorized in the system."""
        address = Web3.to_checksum_address(address)
        institution = self.contract.functions.institutions(address).call()
        return institution[2]  # authorized field
    
    def get_current_round(self) -> int:
        """Get the current training round ID."""
        return self.contract.functions.currentRound().call()
    
    def get_training_round_info(self, round_id: int) -> Dict[str, Any]:
        """
        Get information about a training round.
        
        Returns:
            Dict with round information
        """
        try:
            round_info = self.contract.functions.trainingRounds(round_id).call()
            return {
                'roundId': round_info[0],
                'startTime': round_info[1],
                'endTime': round_info[2],
                'globalModelHash': round_info[3],
                'completed': round_info[4]
            }
        except (ContractLogicError, IndexError):
            return {}
    
    def get_participants(self, round_id: int) -> List[str]:
        """Get the list of participants for a training round."""
        try:
            participant_count = self.contract.functions.getParticipantCount(round_id).call()
            participants = []
            
            for i in range(participant_count):
                participant = self.contract.functions.getParticipantAtIndex(round_id, i).call()
                participants.append(participant)
                
            return participants
        except (ContractLogicError, IndexError):
            return []
    
    def get_model_update(self, round_id: int, institution_address: str) -> Dict[str, Any]:
        """
        Get model update details for a specific institution in a round.
        
        Returns:
            Dict with update information
        """
        institution_address = Web3.to_checksum_address(institution_address)
        
        try:
            update = self.contract.functions.getModelUpdateByInstitution(
                round_id, 
                institution_address
            ).call()
            
            return {
                'ipfsHash': update[0],
                'metrics': update[1],
                'timestamp': update[2]
            }
        except (ContractLogicError, IndexError):
            return {}