"""
GA-Stacking Reward System for Federated Learning with Blockchain Integration.
This module handles the evaluation and reward distribution for GA-Stacking ensembles.
"""

import logging
import json
import numpy as np
from datetime import datetime, timezone
import time


class GAStackingRewardSystem:
    """
    A reward system specifically designed for GA-Stacking federated learning.
    Integrates with the blockchain to track and distribute rewards based on
    the quality of GA-Stacking ensembles.
    """
    
    def __init__(self, blockchain_connector, config_path="config/ga_reward_config.json"):
        """
        Initialize the GA-Stacking reward system.
        
        Args:
            blockchain_connector: BlockchainConnector instance
            config_path: Path to configuration file
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger('GAStackingRewardSystem')
        
        # Load GA-specific configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded GA-Stacking reward configuration from {config_path}")
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            # Default configuration
            self.config = {
                "metric_weights": {
                    "ensemble_accuracy": 0.40,
                    "diversity_score": 0.20,
                    "generalization_score": 0.20,
                    "convergence_rate": 0.10,
                    "avg_base_model_score": 0.10
                },
                "reward_scaling": {
                    "base_amount": 0.1,  # ETH per round
                    "increment_per_round": 0.02,  # Increase each round
                    "accuracy_bonus_threshold": 0.9,  # Bonus for >90% accuracy
                    "bonus_multiplier": 1.5  # 50% bonus for high accuracy
                }
            }
            self.logger.info("Using default GA-Stacking reward configuration")
    
    def start_training_round(self, round_number):
        """
        Start a new GA-Stacking training round with an appropriate reward pool.
        
        Args:
            round_number: Current federated learning round number
            
        Returns:
            tuple: (success, tx_hash)
        """
        try:
            # Calculate dynamic reward amount based on round number
            base_amount = self.config["reward_scaling"]["base_amount"]
            increment = self.config["reward_scaling"]["increment_per_round"]
            
            # Reward amount increases with round number
            reward_amount = base_amount + (round_number - 1) * increment
            
            # Check if round is already funded
            pool_info = self.get_reward_pool_info(round_number)
            
            # Only fund if not already funded
            if pool_info['total_eth'] == 0:
                self.logger.info(f"Funding round {round_number} with {reward_amount} ETH")
                
                # Try different methods to fund the pool
                tx_hash = None
                
                try:
                    # Method 1: Use blockchain connector's method if available
                    if hasattr(self.blockchain, 'fund_round_reward_pool'):
                        tx_hash = self.blockchain.fund_round_reward_pool(round_number, reward_amount)
                    # Method 2: Use RewardManager if available
                    elif hasattr(self.blockchain, 'reward_manager') and hasattr(self.blockchain.reward_manager, 'fund_reward_pool'):
                        success, tx_hash = self.blockchain.reward_manager.fund_reward_pool(round_number, reward_amount)
                    # Method 3: Direct contract call
                    else:
                        # Convert ETH to wei if web3 is available
                        if hasattr(self.blockchain, 'web3'):
                            reward_wei = self.blockchain.web3.toWei(reward_amount, 'ether')
                        else:
                            # Simple conversion (1 ETH = 10^18 wei)
                            reward_wei = int(reward_amount * 10**18)
                            
                        # Get admin address
                        admin_address = getattr(self.blockchain, 'admin_address', 
                                    getattr(self.blockchain, 'account_address', 
                                    getattr(self.blockchain, 'owner_address', None)))
                        
                        if admin_address:
                            tx = self.blockchain.contract.functions.fundRoundRewardPool(round_number).transact({
                                'from': admin_address,
                                'value': reward_wei
                            })
                            tx_hash = tx.hex() if hasattr(tx, 'hex') else tx
                except Exception as e:
                    self.logger.error(f"Error during fund transaction: {e}")
                    
                if tx_hash:
                    self.logger.info(f"Successfully funded round {round_number} with {reward_amount} ETH")
                    return True, tx_hash
                else:
                    self.logger.error(f"Transaction failed for funding round {round_number}")
                    return False, None
            else:
                self.logger.info(f"Round {round_number} already funded with {pool_info['total_eth']} ETH")
                return True, None
                
        except Exception as e:
            self.logger.error(f"Error funding reward pool for round {round_number}: {e}")
            return False, None
    
    def record_client_contribution(self, client_address, ipfs_hash, metrics, round_number):
        """
        Record a client's GA-Stacking contribution on the blockchain.
        
        Args:
            client_address: Client's Ethereum address
            ipfs_hash: IPFS hash of the client's model
            metrics: Evaluation metrics dict with GA-Stacking measures
            round_number: Current FL round number
            
        Returns:
            tuple: (success, recorded_score, transaction_hash)
        """
        try:
            # Ensure we have a valid score
            if 'final_score' not in metrics:
                # Calculate score from individual metrics
                weights = self.config["metric_weights"]
                weighted_score = (
                    metrics.get('ensemble_accuracy', 0.0) * weights['ensemble_accuracy'] +
                    metrics.get('diversity_score', 0.0) * weights['diversity_score'] +
                    metrics.get('generalization_score', 0.0) * weights['generalization_score'] +
                    metrics.get('convergence_rate', 0.5) * weights['convergence_rate'] +
                    metrics.get('avg_base_model_score', 0.0) * weights['avg_base_model_score']
                )
                
                # Apply bonus for exceptional accuracy
                bonus_threshold = self.config["reward_scaling"]["accuracy_bonus_threshold"]
                if metrics.get('ensemble_accuracy', 0.0) > bonus_threshold:
                    bonus_multiplier = self.config["reward_scaling"]["bonus_multiplier"]
                    additional_score = (metrics['ensemble_accuracy'] - bonus_threshold) * bonus_multiplier
                    weighted_score += additional_score
                
                # Convert to integer score (0-10000)
                metrics['final_score'] = int(min(1.0, weighted_score) * 10000)
            
            score = metrics['final_score']
            
            # Record on blockchain
            try:
                tx_hash = self.blockchain.record_contribution(
                    client_address=client_address,
                    round_num=round_number,
                    ipfs_hash=ipfs_hash,
                    accuracy=score  # Pass the final_score as accuracy
                )
                
                if tx_hash:
                    self.logger.info(f"Recorded GA-Stacking contribution from {client_address} with score {score}")
                    return True, score, tx_hash
                else:
                    self.logger.error(f"Failed to record GA-Stacking contribution for {client_address}")
                    return False, 0, None
            except Exception as tx_error:
                self.logger.error(f"Transaction error: {tx_error}")
                return False, 0, None
                
        except Exception as e:
            self.logger.error(f"Error recording GA-Stacking contribution: {e}")
            return False, 0, None
    
    def finalize_round_and_allocate_rewards(self, round_number):
        """
        Finalize a GA-Stacking training round and allocate rewards to contributors.
        
        Args:
            round_number: Federated learning round number
            
        Returns:
            tuple: (success, allocated_amount)
        """
        try:
            # Check if pool has any funds before attempting to finalize
            pool_info = self.get_reward_pool_info(round_number)
            
            if pool_info['total_eth'] == 0:
                self.logger.error(f"Cannot finalize reward pool for round {round_number}: Empty pool")
                return False, 0
                    
            if pool_info['is_finalized']:
                self.logger.warning(f"Reward pool for round {round_number} already finalized")
                
                # Check if there are any remaining rewards
                if pool_info['remaining_eth'] <= 0:
                    self.logger.warning(f"No remaining rewards for round {round_number}")
                    return True, pool_info['allocated_eth']  # Return already allocated amount as success
                    
                # Even if already finalized, we can try to allocate rewards
                try:
                    # Implement retry logic for allocation
                    success, allocated = self._allocate_rewards_with_retry(round_number)
                    return success, allocated
                except Exception as alloc_err:
                    self.logger.error(f"Error allocating rewards for already finalized round: {alloc_err}")
                    return False, 0
            
            # First finalize the reward pool if not already finalized
            tx_hash = self.blockchain.finalize_round_reward_pool(round_number)
            
            if not tx_hash:
                self.logger.error(f"Failed to finalize reward pool for round {round_number}")
                return False, 0
            
            self.logger.info(f"Finalized reward pool for round {round_number}")
            
            # Add a small delay to ensure finalization is processed
            time.sleep(1)
            
            # Then allocate rewards with retry
            return self._allocate_rewards_with_retry(round_number)
                    
        except Exception as e:
            self.logger.error(f"Error in reward finalization process: {e}")
            return False, 0
        
    def _allocate_rewards_with_retry(self, round_number, max_retries=5, initial_delay=1):
        """Helper method to allocate rewards with retry logic."""
        retry_delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                tx_hash = self.blockchain.allocate_rewards_for_round(round_number)
                
                if tx_hash:
                    # Get pool info to determine allocated amount
                    updated_pool = self.get_reward_pool_info(round_number)
                    allocated_eth = updated_pool['allocated_eth']
                    
                    self.logger.info(f"Successfully allocated {allocated_eth} ETH rewards for round {round_number}")
                    return True, allocated_eth
                else:
                    self.logger.error(f"Failed to allocate rewards for round {round_number}")
                    
                    # Check if all rewards have been allocated
                    pool_info = self.get_reward_pool_info(round_number)
                    if pool_info['remaining_eth'] <= 0:
                        self.logger.warning(f"All rewards have been allocated for round {round_number}")
                        return True, pool_info['allocated_eth']
                        
                    # If there are still rewards, but transaction failed, retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                        
                    return False, 0
                    
            except Exception as e:
                err_str = str(e)
                
                # Check for specific error messages
                if "No available rewards" in err_str or "revert No available rewards" in err_str:
                    self.logger.warning(f"All rewards have been allocated for round {round_number}")
                    # Get current allocation
                    current_pool = self.get_reward_pool_info(round_number)
                    return True, current_pool['allocated_eth']
                
                # Log and retry for other errors
                self.logger.warning(f"Transaction failed (attempt {attempt+1}/{max_retries}), retrying in {retry_delay:.1f} seconds: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                    
                self.logger.error(f"Failed to allocate rewards after {max_retries} attempts")
                return False, 0
                
        return False, 0  # Should not reach here, but added as a fallback
    
    def _has_method(self, obj, method_name):
        """Check if an object has a specific method."""
        return callable(getattr(obj, method_name, None))
    
    def get_reward_pool_info(self, round_number):
        """
        Get information about a round's reward pool.
        
        Args:
            round_number: The federated learning round number
            
        Returns:
            dict: Reward pool information
        """
        try:
            if self._has_method(self.blockchain, 'get_round_reward_pool'):
                pool_info = self.blockchain.get_round_reward_pool(round_number)
                return {
                    'round': round_number,
                    'total_eth': pool_info['total_amount'],
                    'allocated_eth': pool_info['allocated_amount'],
                    'remaining_eth': pool_info['remaining_amount'],
                    'is_finalized': pool_info['is_finalized']
                }
            else:
                # Fallback to direct contract call
                pool_info = self.blockchain.contract.functions.getRoundRewardPool(round_number).call()
                total_amount, allocated_amount, remaining_amount, is_finalized = pool_info
                
                # Convert wei to ETH if web3 is available
                try:
                    web3 = getattr(self.blockchain, 'web3', None)
                    if web3:
                        total_eth = web3.from_wei(total_amount, 'ether')
                        allocated_eth = web3.from_wei(allocated_amount, 'ether')
                        remaining_eth = web3.from_wei(remaining_amount, 'ether')
                    else:
                        # Simple conversion if web3 not available (1 ETH = 10^18 wei)
                        total_eth = total_amount / 10**18
                        allocated_eth = allocated_amount / 10**18
                        remaining_eth = remaining_amount / 10**18
                except:
                    # If conversion fails, return raw values
                    total_eth = total_amount
                    allocated_eth = allocated_amount
                    remaining_eth = remaining_amount
                
                return {
                    'round': round_number,
                    'total_eth': total_eth,
                    'allocated_eth': allocated_eth,
                    'remaining_eth': remaining_eth,
                    'is_finalized': is_finalized
                }
            
        except Exception as e:
            self.logger.error(f"Error getting reward pool info: {e}")
            return {
                'round': round_number,
                'total_eth': 0,
                'allocated_eth': 0,
                'remaining_eth': 0,
                'is_finalized': False
            }
    
    def get_round_contributions(self, round_number, offset=0, limit=100):
        """
        Get all contributions for a specific round with pagination.
        
        Args:
            round_number: The federated learning round number
            offset: Starting index for pagination
            limit: Maximum number of records to return
            
        Returns:
            list: List of contribution records
        """
        try:
            # Use blockchain connector's method
            if hasattr(self.blockchain, 'get_round_contributions'):
                return self.blockchain.get_round_contributions(round_number, offset, limit)
            else:
                # Direct contract call
                records = self.blockchain.contract.functions.getRoundContributions(
                    round_number, 
                    offset,
                    limit
                ).call()
                
                clients, accuracies, scores, rewarded = records
                
                # Format the results as a list of dictionaries
                contributions = []
                for i in range(len(clients)):
                    if clients[i] != '0x0000000000000000000000000000000000000000':  # Skip empty entries
                        contributions.append({
                            'client_address': clients[i],
                            'accuracy': accuracies[i] / 10000.0,  # Convert back to percentage
                            'score': scores[i],
                            'rewarded': rewarded[i]
                        })
                
                return contributions
            
        except Exception as e:
            self.logger.error(f"Error getting round contributions: {e}")
            return []
    
    def get_round_contributions_with_metrics(self, round_number):
        """
        Get all contributions for a round with detailed metrics.
        
        Args:
            round_number: Federated learning round number
            
        Returns:
            dict: Detailed contribution records with statistics
        """
        try:
            # Call with correct arguments (round_number, offset, limit)
            contributions = self.get_round_contributions(round_number, 0, 100)
            
            # Process contributions to create summary
            if contributions:
                # Calculate average score
                scores = [c.get('score', 0) for c in contributions]
                if not scores:
                    return {'contributions': [], 'summary': {}}
                    
                avg_score = sum(scores) / len(scores)
                
                # Calculate distribution statistics
                score_std = np.std(scores) if len(scores) > 1 else 0
                score_min = min(scores) if scores else 0
                score_max = max(scores) if scores else 0
                
                # Add analysis to each contribution
                for contribution in contributions:
                    # Calculate relative performance (percentile)
                    contribution['percentile'] = sum(1 for s in scores if s <= contribution.get('score', 0)) / len(scores)
                    
                    # Calculate z-score (how many standard deviations from mean)
                    if score_std > 0:
                        contribution['z_score'] = (contribution.get('score', 0) - avg_score) / score_std
                    else:
                        contribution['z_score'] = 0
                
                # Add summary statistics
                contributions_with_stats = {
                    'contributions': contributions,
                    'summary': {
                        'count': len(contributions),
                        'avg_score': avg_score,
                        'std_deviation': score_std,
                        'min_score': score_min,
                        'max_score': score_max,
                        'score_range': score_max - score_min if len(scores) > 1 else 0
                    }
                }
                
                return contributions_with_stats
            
            return {'contributions': [], 'summary': {}}
        except Exception as e:
            self.logger.error(f"Error in get_round_contributions_with_metrics: {e}")
            return {'contributions': [], 'summary': {}}
    
    def get_client_rewards(self, client_address):
        """
        Get available rewards for a client.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            float: Available rewards in ETH
        """
        try:
            if hasattr(self.blockchain, 'get_available_rewards'):
                return self.blockchain.get_available_rewards(client_address)
            elif hasattr(self.blockchain.contract.functions, 'getAvailableRewards'):
                rewards_wei = self.blockchain.contract.functions.getAvailableRewards(client_address).call()
                # Convert wei to ETH if web3 is available
                try:
                    web3 = getattr(self.blockchain, 'web3', None)
                    if web3:
                        return float(web3.from_wei(rewards_wei, 'ether'))
                    else:
                        # Simple conversion if web3 not available (1 ETH = 10^18 wei)
                        return float(rewards_wei / 10**18)
                except:
                    return float(rewards_wei)
            else:
                self.logger.warning("Method to get client rewards not found")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error getting client rewards: {e}")
            return 0.0