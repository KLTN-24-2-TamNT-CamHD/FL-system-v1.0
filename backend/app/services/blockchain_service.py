from web3 import Web3
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from eth_account import Account
from app.config import settings
import json

class BlockchainService:
    def __init__(self):
        self._w3 = None
        self._contract = None
        self.admin_account = None
        self.timestamp = "2025-03-31 16:03:51"
        self.admin = "dinhcam89"
        
    @property
    def w3(self):
        if self._w3 is None:
            try:
                self._w3 = Web3(Web3.HTTPProvider(settings.WEB3_PROVIDER_URI))
                if not self._w3.is_connected():
                    raise ConnectionError("Failed to connect to Ethereum network")
            except Exception as e:
                print(f"Blockchain connection error: {str(e)}")
                self._w3 = None
        return self._w3
        
    @property
    def contract(self):
        if self._contract is None and self.w3 is not None:
            try:
                contract_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../contracts/FraudDetectionFederated.json'
                )
                
                with open(contract_path) as f:
                    contract_json = json.load(f)
                    contract_abi = contract_json['abi']
                
                self._contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(settings.CONTRACT_ADDRESS),
                    abi=contract_abi
                )
            except Exception as e:
                print(f"Contract initialization error: {str(e)}")
                self._contract = None
        return self._contract

    def is_connected(self) -> bool:
        """Check if connected to blockchain network"""
        try:
            return self.w3 is not None and self.w3.is_connected()
        except:
            return False

    async def get_current_round(self) -> int:
        """Get current round with error handling"""
        try:
            if not self.is_connected():
                return -1
            return await self.contract.functions.currentRound().call()
        except Exception as e:
            print(f"Error getting current round: {str(e)}")
            return -1

    async def register_institution(self, institution_address: str, name: str) -> str:
        tx = self.contract.functions.registerInstitution(
            institution_address,
            name
        ).build_transaction({
            'from': self.w3.eth.accounts[0],
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0]),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, settings.PRIVATE_KEY)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.to_hex(tx_hash)

    async def initiate_training_round(self) -> str:
        tx = self.contract.functions.initiateTrainingRound().build_transaction({
            'from': self.w3.eth.accounts[0],
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0]),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, settings.PRIVATE_KEY)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.to_hex(tx_hash)

    async def submit_model_update(
        self,
        round_id: int,
        ipfs_hash: str,
        metrics: str
    ) -> str:
        tx = self.contract.functions.submitModelUpdate(
            round_id,
            ipfs_hash,
            metrics
        ).build_transaction({
            'from': self.w3.eth.accounts[0],
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0]),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, settings.PRIVATE_KEY)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.to_hex(tx_hash)

    async def complete_round(self, round_id: int, global_model_hash: str) -> str:
        tx = self.contract.functions.completeRound(
            round_id,
            global_model_hash
        ).build_transaction({
            'from': self.w3.eth.accounts[0],
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0]),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, settings.PRIVATE_KEY)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.to_hex(tx_hash)

    async def submit_evaluation(
        self,
        round_id: int,
        loss: int,
        accuracy: int,
        auc: int,
        precision: int,
        recall: int
    ) -> str:
        tx = self.contract.functions.submitEvaluation(
            round_id,
            loss,
            accuracy,
            auc,
            precision,
            recall
        ).build_transaction({
            'from': self.w3.eth.accounts[0],
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.accounts[0]),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, settings.PRIVATE_KEY)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.to_hex(tx_hash)

    async def get_training_round_info(self, round_id: int) -> Dict:
        round_info = await self.contract.functions.trainingRounds(round_id).call()
        participant_count = await self.contract.functions.getParticipantCount(round_id).call()
        
        participants = []
        for i in range(participant_count):
            participant = await self.contract.functions.getParticipantAtIndex(round_id, i).call()
            update = await self.contract.functions.getModelUpdateByInstitution(round_id, participant).call()
            participants.append({
                "address": participant,
                "update": {
                    "ipfs_hash": update[0],
                    "metrics": update[1],
                    "timestamp": update[2]
                }
            })
        
        return {
            "round_id": round_id,
            "start_time": round_info[1],
            "end_time": round_info[2],
            "global_model_hash": round_info[3],
            "completed": round_info[4],
            "participants": participants
        }

blockchain_service = BlockchainService()