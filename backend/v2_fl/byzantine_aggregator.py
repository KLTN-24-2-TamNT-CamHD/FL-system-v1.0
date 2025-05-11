"""
Byzantine-robust ensemble aggregation strategy for federated learning with GA-Stacking.
Handles aggregation of ensemble models from multiple clients with protection against malicious updates.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict

import torch
import torch.nn as nn

from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

logger = logging.getLogger("Robust-Ensemble-Aggregation")


class ByzantineRobustEnsembleAggregator:
    """Aggregates ensemble models with Byzantine-robust methods in federated learning."""
    
    def __init__(
        self, 
        device: str = "cpu",
        robust_method: str = "trimmed_mean",
        trim_ratio: float = 0.2,
        z_score_threshold: float = 2.0,
        min_client_contribution: int = 3,
        reputation_threshold: float = 0.6,
    ):
        """
        Initialize the Byzantine-robust ensemble aggregator.
        
        Args:
            device: Device to use for computations
            robust_method: Aggregation method to use ("trimmed_mean", "median", "krum", "bulyan")
            trim_ratio: Ratio of clients to trim when using trimmed mean (0.0-0.5)
            z_score_threshold: Threshold for z-score outlier detection
            min_client_contribution: Minimum number of clients needed for aggregation
            reputation_threshold: Minimum reputation score for client contribution acceptance (0.0-1.0)
        """
        self.device = torch.device(device)
        self.robust_method = robust_method
        self.trim_ratio = max(0.0, min(0.5, trim_ratio))  # Keep between 0 and 0.5
        self.z_score_threshold = z_score_threshold
        self.min_client_contribution = min_client_contribution
        self.reputation_threshold = reputation_threshold
        
        # Client reputation tracking
        self.client_reputations = {}  # Maps client_id to reputation score (0-1)
    
    def deserialize_ensemble(self, parameters: List[np.ndarray]) -> Dict[str, Any]:
        """
        Deserialize ensemble model from parameters.
        
        Args:
            parameters: List of parameter arrays
            
        Returns:
            Deserialized ensemble state
        """
        if len(parameters) != 1 or parameters[0].dtype != np.uint8:
            logger.warning("Parameters don't appear to be a serialized ensemble")
            return None
        
        try:
            # Convert bytes to ensemble state
            ensemble_bytes = parameters[0].tobytes()
            ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
            return ensemble_state
        except Exception as e:
            logger.error(f"Failed to deserialize ensemble state: {e}")
            return None
    
    def serialize_ensemble(self, ensemble_state: Dict[str, Any]) -> List[np.ndarray]:
        """
        Serialize ensemble state to parameters.
        
        Args:
            ensemble_state: Ensemble state dictionary
            
        Returns:
            List of parameter arrays
        """
        try:
            # Convert ensemble state to bytes
            ensemble_bytes = json.dumps(ensemble_state).encode('utf-8')
            return [np.frombuffer(ensemble_bytes, dtype=np.uint8)]
        except Exception as e:
            logger.error(f"Failed to serialize ensemble state: {e}")
            return None
    
    def detect_outliers(self, ensembles: List[Dict[str, Any]]) -> List[int]:
        """
        Detect outlier ensembles using statistical methods.
        
        Args:
            ensembles: List of ensemble state dictionaries
            
        Returns:
            List of indices of detected outliers
        """
        if len(ensembles) <= self.min_client_contribution:
            # Not enough ensembles to detect outliers
            return []
        
        # Extract model weights as feature vectors for outlier detection
        feature_vectors = []
        
        for ensemble in ensembles:
            # Flatten all weights and parameters into a single feature vector
            feature_vector = []
            
            # Add ensemble weights
            feature_vector.extend(ensemble["weights"])
            
            # For each model, extract numeric parameters that can be compared
            for state_dict in ensemble["model_state_dicts"]:
                for key, value in state_dict.items():
                    # Skip non-numeric parameters
                    if isinstance(value, (int, float)):
                        feature_vector.append(float(value))
                    elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                        feature_vector.extend([float(x) for x in value])
            
            feature_vectors.append(np.array(feature_vector))
        
        # Detect outliers using z-score method
        outliers = []
        
        # If feature vectors vary in length, use a different approach
        if len(set(len(v) for v in feature_vectors)) != 1:
            logger.warning("Feature vectors have different lengths, using pairwise distance for outlier detection")
            # Calculate pairwise distances between ensembles
            n_ensembles = len(ensembles)
            distances = np.zeros((n_ensembles, n_ensembles))
            
            for i in range(n_ensembles):
                for j in range(i+1, n_ensembles):
                    # Calculate distance between model names
                    names_i = set(ensembles[i]["model_names"])
                    names_j = set(ensembles[j]["model_names"])
                    name_similarity = len(names_i.intersection(names_j)) / max(1, len(names_i.union(names_j)))
                    
                    # Weight differences based on name similarity
                    weight_diff = np.mean(np.abs(np.array(ensembles[i]["weights"]) - np.array(ensembles[j]["weights"])))
                    
                    # Combined distance
                    distance = weight_diff * (2.0 - name_similarity)
                    distances[i, j] = distance
                    distances[j, i] = distance
            
            # Calculate average distance to other clients
            avg_distances = np.sum(distances, axis=1) / (n_ensembles - 1)
            
            # Calculate z-scores of distances
            mean_dist = np.mean(avg_distances)
            std_dist = np.std(avg_distances) + 1e-10  # Avoid division by zero
            z_scores = np.abs((avg_distances - mean_dist) / std_dist)
            
            # Flag outliers based on z-score threshold
            outliers = [i for i, z in enumerate(z_scores) if z > self.z_score_threshold]
            
        else:
            # Convert to numpy array for easier calculations
            features = np.array(feature_vectors)
            
            # Calculate mean and std for each feature
            mean_features = np.mean(features, axis=0)
            std_features = np.std(features, axis=0) + 1e-10  # Avoid division by zero
            
            # Calculate z-scores for each client's features
            z_scores = np.abs((features - mean_features) / std_features)
            
            # Take mean z-score for each client across all features
            mean_z_scores = np.mean(z_scores, axis=1)
            
            # Flag outliers based on z-score threshold
            outliers = [i for i, z in enumerate(mean_z_scores) if z > self.z_score_threshold]
        
        logger.info(f"Detected {len(outliers)} outliers out of {len(ensembles)} ensembles")
        return outliers
    
    def update_client_reputations(
        self, 
        client_ids: List[str], 
        outlier_indices: List[int],
        evaluation_metrics: Optional[List[Dict[str, float]]] = None
    ):
        """
        Update reputation scores for clients based on outlier detection and metrics.
        
        Args:
            client_ids: List of client IDs
            outlier_indices: Indices of clients detected as outliers
            evaluation_metrics: Optional list of performance metrics for each client
        """
        for i, client_id in enumerate(client_ids):
            # Initialize reputation if needed
            if client_id not in self.client_reputations:
                self.client_reputations[client_id] = 0.7  # Start with neutral-positive reputation
            
            current_reputation = self.client_reputations[client_id]
            
            # Penalize outliers
            if i in outlier_indices:
                # Reduce reputation for outliers, but not too harshly in one step
                new_reputation = max(0.0, current_reputation - 0.2)
                logger.info(f"Client {client_id} detected as outlier, reputation: {current_reputation:.2f} -> {new_reputation:.2f}")
            else:
                # Slightly increase reputation for non-outliers
                new_reputation = min(1.0, current_reputation + 0.05)
            
            # Further adjust based on evaluation metrics if available
            if evaluation_metrics and i < len(evaluation_metrics):
                metrics = evaluation_metrics[i]
                
                # Use accuracy or similar metric if available
                if "accuracy" in metrics:
                    accuracy = metrics["accuracy"]
                    # Adjust reputation based on accuracy
                    if accuracy < 0.5:  # Poor performance
                        new_reputation = max(0.0, new_reputation - 0.1)
                    elif accuracy > 0.8:  # Good performance
                        new_reputation = min(1.0, new_reputation + 0.1)
                
                # Could check other metrics here...
            
            # Update reputation with an exponential moving average
            self.client_reputations[client_id] = 0.8 * current_reputation + 0.2 * new_reputation
    
    def aggregate_ensembles(
        self, 
        ensembles: List[Dict[str, Any]], 
        weights: List[float],
        client_ids: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], List[int], List[str]]:
        """
        Aggregate multiple ensemble models with Byzantine-robust methods.
        
        Args:
            ensembles: List of ensemble state dictionaries
            weights: Weights for each ensemble
            client_ids: Optional list of client IDs
            
        Returns:
            Tuple of (aggregated ensemble state, indices of trusted ensembles, trusted client IDs)
        """
        if not ensembles:
            logger.error("No ensembles to aggregate")
            return None, [], []
        
        # Use client IDs if provided, otherwise use generic IDs
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(ensembles))]
        
        # Detect outliers
        outlier_indices = self.detect_outliers(ensembles)
        
        # Update client reputations
        self.update_client_reputations(client_ids, outlier_indices)
        
        # Filter out ensembles from low-reputation clients
        trusted_indices = []
        for i, client_id in enumerate(client_ids):
            reputation = self.client_reputations.get(client_id, 0.0)
            if reputation >= self.reputation_threshold and i not in outlier_indices:
                trusted_indices.append(i)
        
        # Check if we have enough trusted ensembles
        if len(trusted_indices) < self.min_client_contribution:
            logger.warning(f"Only {len(trusted_indices)} trusted ensembles, which is below the minimum of {self.min_client_contribution}")
            # If not enough trusted ensembles, use all non-outliers
            trusted_indices = [i for i in range(len(ensembles)) if i not in outlier_indices]
            
            # If still not enough, use the ensembles with highest reputation
            if len(trusted_indices) < self.min_client_contribution and len(ensembles) >= self.min_client_contribution:
                # Sort clients by reputation
                client_reps = [(i, self.client_reputations.get(cid, 0.0)) for i, cid in enumerate(client_ids)]
                client_reps.sort(key=lambda x: x[1], reverse=True)
                # Take top clients to meet minimum
                trusted_indices = [i for i, _ in client_reps[:self.min_client_contribution]]
        
        # If still not enough ensembles, log warning but continue with what we have
        if len(trusted_indices) < self.min_client_contribution:
            logger.warning(f"Proceeding with only {len(trusted_indices)} ensembles, below minimum of {self.min_client_contribution}")
        
        # Select trusted ensembles and adjust weights
        trusted_ensembles = [ensembles[i] for i in trusted_indices]
        trusted_weights = [weights[i] for i in trusted_indices]
        trusted_client_ids = [client_ids[i] for i in trusted_indices]
        
        # Normalize trusted weights
        total_weight = sum(trusted_weights)
        if total_weight == 0:
            logger.warning("Total weight is zero, using equal weights")
            trusted_weights = [1.0/len(trusted_ensembles)] * len(trusted_ensembles)
        else:
            trusted_weights = [w/total_weight for w in trusted_weights]
        
        # Apply robust aggregation method to combine models
        aggregated_ensemble = self._robust_aggregate_ensembles(trusted_ensembles, trusted_weights)
        
        return aggregated_ensemble, trusted_indices, trusted_client_ids
    
    def _robust_aggregate_ensembles(
        self, 
        ensembles: List[Dict[str, Any]], 
        weights: List[float]
    ) -> Dict[str, Any]:
        """
        Apply Byzantine-robust aggregation method to ensembles.
        
        Args:
            ensembles: List of ensemble state dictionaries
            weights: Weights for each ensemble
            
        Returns:
            Aggregated ensemble state
        """
        # Get all unique model names across all ensembles
        all_model_names = set()
        for ensemble in ensembles:
            all_model_names.update(ensemble["model_names"])
        
        # Create mapping from model name to index for each ensemble
        name_to_idx = []
        for ensemble in ensembles:
            mapping = {name: idx for idx, name in enumerate(ensemble["model_names"])}
            name_to_idx.append(mapping)
        
        # Collect all model state dicts by model name
        model_states_by_name = defaultdict(list)
        model_weights_by_name = defaultdict(list)
        ensemble_indices_by_name = defaultdict(list)
        
        for i, ensemble in enumerate(ensembles):
            ensemble_weight = weights[i]
            for name, idx in name_to_idx[i].items():
                model_states_by_name[name].append(ensemble["model_state_dicts"][idx])
                # Calculate effective weight = ensemble_weight * model_weight
                model_weight = ensemble["weights"][idx] * ensemble_weight
                model_weights_by_name[name].append(model_weight)
                ensemble_indices_by_name[name].append(i)
        
        # Aggregate model state dicts using robust method
        aggregated_models = []
        aggregated_names = []
        aggregated_weights = []
        
        for name in sorted(model_states_by_name.keys()):
            # Get all states for this model
            states = model_states_by_name[name]
            model_weights = model_weights_by_name[name]
            ensemble_indices = ensemble_indices_by_name[name]
            
            # Normalize weights for this model
            total_weight = sum(model_weights)
            if total_weight > 0:
                normalized_weights = [w/total_weight for w in model_weights]
            else:
                normalized_weights = [1.0/len(states)] * len(states)
            
            # Aggregate state dicts using robust method
            aggregated_state = self._robust_aggregate_state_dicts(states, normalized_weights, ensemble_indices)
            
            if aggregated_state:
                aggregated_models.append(aggregated_state)
                aggregated_names.append(name)
                # Use average of model weights across ensembles as starting point
                avg_weight = sum(model_weights) / len(model_weights) if model_weights else 0.0
                aggregated_weights.append(avg_weight)
        
        # Normalize final weights
        total_weight = sum(aggregated_weights)
        if total_weight > 0:
            aggregated_weights = [w/total_weight for w in aggregated_weights]
        else:
            aggregated_weights = [1.0/len(aggregated_models)] * len(aggregated_models)
        
        # Create aggregated ensemble state
        aggregated_ensemble = {
            "weights": aggregated_weights,
            "model_names": aggregated_names,
            "model_state_dicts": aggregated_models
        }
        
        return aggregated_ensemble
    
    def _robust_aggregate_state_dicts(
        self, 
        state_dicts: List[Dict[str, Any]], 
        weights: List[float],
        ensemble_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple state dictionaries using Byzantine-robust methods.
        
        Args:
            state_dicts: List of state dictionaries
            weights: Weight for each state dict
            ensemble_indices: Original ensemble indices for each state dict
            
        Returns:
            Aggregated state dictionary
        """
        # Check if we have anything to aggregate
        if not state_dicts:
            return None
        
        # If only one state dict, just return it
        if len(state_dicts) == 1:
            return state_dicts[0]
        
        # Get all keys from all state dicts
        all_keys = set()
        for sd in state_dicts:
            all_keys.update(sd.keys())
        
        # Initialize aggregated state dict
        aggregated = {}
        
        # Aggregate each parameter
        for key in all_keys:
            # Check which state dicts have this key
            valid_dicts = []
            valid_weights = []
            valid_indices = []
            
            for i, sd in enumerate(state_dicts):
                if key in sd:
                    valid_dicts.append(sd[key])
                    valid_weights.append(weights[i])
                    valid_indices.append(ensemble_indices[i])
            
            # Skip if no valid dicts
            if not valid_dicts:
                continue
            
            # Normalize weights
            total_weight = sum(valid_weights)
            if total_weight > 0:
                normalized_weights = [w/total_weight for w in valid_weights]
            else:
                normalized_weights = [1.0/len(valid_dicts)] * len(valid_dicts)
            
            # Handle string parameters (like "estimator")
            if all(isinstance(param, str) for param in valid_dicts):
                # Take the most common string value
                from collections import Counter
                param_counter = Counter(valid_dicts)
                aggregated[key] = param_counter.most_common(1)[0][0]
                continue
            
            # Handle scalar parameters (int, float)
            if all(isinstance(param, (int, float)) for param in valid_dicts):
                # Apply robust aggregation to numeric values
                values = np.array(valid_dicts)
                
                if self.robust_method == "median":
                    # Median is naturally robust to outliers
                    aggregated[key] = float(np.median(values))
                    # For integers, round to nearest int
                    if all(isinstance(param, int) for param in valid_dicts):
                        aggregated[key] = int(round(aggregated[key]))
                
                elif self.robust_method == "trimmed_mean":
                    # Sort values
                    sorted_idx = np.argsort(values)
                    n_values = len(values)
                    n_trim = int(n_values * self.trim_ratio)
                    
                    # Calculate trimmed mean (exclude lowest and highest values)
                    if n_trim > 0:
                        trimmed_values = values[sorted_idx[n_trim:-n_trim]]
                        trimmed_weights = [normalized_weights[i] for i in sorted_idx[n_trim:-n_trim]]
                        
                        # Re-normalize weights
                        total = sum(trimmed_weights)
                        if total > 0:
                            trimmed_weights = [w/total for w in trimmed_weights]
                        else:
                            trimmed_weights = [1.0/len(trimmed_values)] * len(trimmed_values)
                        
                        # Weighted average of remaining values
                        result = sum(v * w for v, w in zip(trimmed_values, trimmed_weights))
                    else:
                        # If no trimming, use weighted average
                        result = sum(v * w for v, w in zip(values, normalized_weights))
                    
                    aggregated[key] = result
                    # For integers, round to nearest int
                    if all(isinstance(param, int) for param in valid_dicts):
                        aggregated[key] = int(round(result))
                
                elif self.robust_method == "krum":
                    # Krum selects the model closest to its peers
                    n_values = len(values)
                    if n_values <= 2:
                        # Not enough values for Krum, use median
                        aggregated[key] = float(np.median(values))
                    else:
                        # Calculate distance between each pair of values
                        distances = np.zeros((n_values, n_values))
                        for i in range(n_values):
                            for j in range(i+1, n_values):
                                dist = abs(values[i] - values[j])
                                distances[i, j] = dist
                                distances[j, i] = dist
                        
                        # For each value, sum distances to closest n-f-2 values
                        # where f is assumed to be (n-1)/3 Byzantine clients
                        f = max(1, int((n_values - 1) / 3))
                        neighbor_count = n_values - f - 2
                        
                        if neighbor_count <= 0:
                            # Not enough values for proper Krum, use median
                            aggregated[key] = float(np.median(values))
                        else:
                            scores = []
                            for i in range(n_values):
                                # Get distances to other values, sorted
                                dist_i = np.sort(distances[i])
                                # Sum distances to closest neighbors
                                score = np.sum(dist_i[1:neighbor_count+1])  # Skip self (0 distance)
                                scores.append(score)
                            
                            # Select value with smallest score
                            best_idx = np.argmin(scores)
                            aggregated[key] = float(values[best_idx])
                        
                        # For integers, round to nearest int
                        if all(isinstance(param, int) for param in valid_dicts):
                            aggregated[key] = int(round(aggregated[key]))
                
                else:  # Default to weighted average
                    result = sum(v * w for v, w in zip(values, normalized_weights))
                    aggregated[key] = result
                    # For integers, round to nearest int
                    if all(isinstance(param, int) for param in valid_dicts):
                        aggregated[key] = int(round(result))
                
                continue
            
            # Handle list parameters (including nested lists)
            try:
                # Try to convert all parameters to float arrays
                param_arrays = []
                for param in valid_dicts:
                    # Handle potential nested lists (e.g., for coef matrices)
                    if isinstance(param, list) and all(isinstance(item, list) for item in param):
                        # For nested lists, convert each inner list to float
                        param_array = np.array([
                            [float(x) if isinstance(x, (int, float, str)) else x for x in inner]
                            for inner in param
                        ], dtype=np.float64)
                    elif isinstance(param, list):
                        # For flat lists, convert directly to float
                        param_array = np.array([
                            float(x) if isinstance(x, (int, float, str)) else x for x in param
                        ], dtype=np.float64)
                    else:
                        # If it's not a list, create a scalar array
                        param_array = np.array([float(param)], dtype=np.float64)
                    
                    param_arrays.append(param_array)
                
                # Check shapes after conversion
                shapes = [arr.shape for arr in param_arrays]
                if len(set(shapes)) != 1:
                    logger.warning(f"Skipping robust aggregation for parameter {key} due to shape mismatch: {shapes}")
                    # Use the parameter from the client with the highest weight
                    max_weight_idx = normalized_weights.index(max(normalized_weights))
                    aggregated[key] = valid_dicts[max_weight_idx]
                    continue
                
                # Stack arrays for easier processing
                shape = shapes[0]
                stacked_arrays = np.stack(param_arrays, axis=0)
                
                # Apply robust aggregation based on method
                if self.robust_method == "median":
                    # Element-wise median
                    aggregated_param = np.median(stacked_arrays, axis=0)
                
                elif self.robust_method == "trimmed_mean":
                    # Element-wise trimmed mean
                    n_clients = stacked_arrays.shape[0]
                    n_trim = int(n_clients * self.trim_ratio)
                    
                    if n_trim > 0 and n_clients > 2 * n_trim:
                        # Sort along client dimension
                        sorted_arrays = np.sort(stacked_arrays, axis=0)
                        # Take trimmed mean
                        aggregated_param = np.mean(sorted_arrays[n_trim:-n_trim], axis=0)
                    else:
                        # Fall back to regular mean if not enough clients
                        aggregated_param = np.average(stacked_arrays, axis=0, weights=normalized_weights)
                
                elif self.robust_method == "krum":
                    # Multi-Krum for array parameters
                    n_clients = stacked_arrays.shape[0]
                    if n_clients <= 2:
                        # Not enough clients for Krum, use median
                        aggregated_param = np.median(stacked_arrays, axis=0)
                    else:
                        # Calculate pairwise distances between client parameters
                        distances = np.zeros((n_clients, n_clients))
                        for i in range(n_clients):
                            for j in range(i+1, n_clients):
                                # Euclidean distance between parameters
                                dist = np.linalg.norm(stacked_arrays[i] - stacked_arrays[j])
                                distances[i, j] = dist
                                distances[j, i] = dist
                        
                        # For each client, calculate score based on distances to closest peers
                        f = max(1, int((n_clients - 1) / 3))  # Assumed Byzantine clients
                        neighbor_count = n_clients - f - 2
                        
                        if neighbor_count <= 0:
                            # Not enough clients for proper Krum, use median
                            aggregated_param = np.median(stacked_arrays, axis=0)
                        else:
                            scores = []
                            for i in range(n_clients):
                                # Get distances to other clients, sorted
                                dist_i = np.sort(distances[i])
                                # Sum distances to closest neighbors
                                score = np.sum(dist_i[1:neighbor_count+1])  # Skip self (0 distance)
                                scores.append(score)
                            
                            # Select client with smallest score
                            best_idx = np.argmin(scores)
                            aggregated_param = stacked_arrays[best_idx]
                
                else:  # Default to weighted average
                    # Element-wise weighted average
                    aggregated_param = np.average(stacked_arrays, axis=0, weights=normalized_weights)
                
                # Convert back to list or scalar
                if len(shape) > 0:
                    aggregated[key] = aggregated_param.tolist()
                else:
                    # It's a scalar array, return as float
                    aggregated[key] = float(aggregated_param)
                
            except (ValueError, TypeError) as e:
                # If we can't convert to float arrays, this might be a non-numeric parameter
                logger.warning(f"Cannot apply robust aggregation to parameter {key} numerically: {e}")
                # Use the parameter from the client with the highest weight
                max_weight_idx = normalized_weights.index(max(normalized_weights))
                aggregated[key] = valid_dicts[max_weight_idx]
        
        return aggregated
    
    def aggregate_fit_results(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        weights: List[float]
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        """
        Aggregate fit results from multiple clients with Byzantine robustness.
        
        Args:
            results: List of (client, fit_res) tuples
            weights: Weight for each result based on sample count
            
        Returns:
            Tuple of (parameters, metrics)
        """
        # Extract parameters and ensembles
        ensembles = []
        client_weights = []
        client_ids = []
        
        for i, (client, fit_res) in enumerate(results):
            try:
                # Convert parameters to ndarrays
                params = parameters_to_ndarrays(fit_res.parameters)
                
                # Deserialize ensemble
                ensemble = self.deserialize_ensemble(params)
                
                if ensemble:
                    ensembles.append(ensemble)
                    client_weights.append(weights[i])
                    client_ids.append(client.cid)
                else:
                    logger.warning(f"Client {client.cid} did not return a valid ensemble")
            except Exception as e:
                logger.error(f"Error processing result from client {client.cid}: {e}")
        
        # Check if we have any valid ensembles
        if not ensembles:
            logger.error("No valid ensembles received from clients")
            return None, {"error": "no_valid_ensembles"}
        
        # Aggregate ensembles using robust method
        logger.info(f"Robust aggregation of {len(ensembles)} ensembles using method: {self.robust_method}")
        aggregated_ensemble, trusted_indices, trusted_client_ids = self.aggregate_ensembles(
            ensembles, client_weights, client_ids
        )
        
        # Serialize aggregated ensemble
        parameters = self.serialize_ensemble(aggregated_ensemble)
        
        # Add metrics about the aggregation process
        trusted_pct = 100 * len(trusted_indices) / len(ensembles) if ensembles else 0
        metrics = {
            "num_ensembles": len(ensembles),
            "num_trusted_ensembles": len(trusted_indices),
            "trusted_percentage": trusted_pct,
            "num_models": len(aggregated_ensemble["model_names"]),
            "model_names": ",".join(aggregated_ensemble["model_names"]),
            "ensemble_weights": aggregated_ensemble["weights"],
            "aggregation_method": self.robust_method,
            "trusted_clients": ",".join(trusted_client_ids)
        }
        
        # Add per-client reputation scores to metrics
        for client_id in client_ids:
            reputation = self.client_reputations.get(client_id, 0.0)
            metrics[f"reputation_{client_id}"] = reputation
        
        return ndarrays_to_parameters(parameters), metrics
    
    def aggregate_evaluate_results(
        self,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        weights: List[float]
    ) -> Tuple[float, Dict[str, Scalar]]:
        """
        Aggregate evaluation results from multiple clients with Byzantine robustness.
        
        Args:
            results: List of (client, evaluate_res) tuples
            weights: Weight for each result based on sample count
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Extract client IDs and metrics for reputation updates
        client_ids = []
        evaluation_metrics = []
        
        for client, eval_res in results:
            client_ids.append(client.cid)
            metrics_dict = {}
            for key, value in eval_res.metrics.items():
                if isinstance(value, (int, float)):
                    metrics_dict[key] = value
            evaluation_metrics.append(metrics_dict)
        
        # Use evaluation metrics to update client reputations
        # No outlier detection here, so pass empty list
        self.update_client_reputations(client_ids, [], evaluation_metrics)
        
        # Mark suspicious clients based on metrics
        suspicious_indices = []
        for i, metrics in enumerate(evaluation_metrics):
            if "accuracy" in metrics:
                # Flag clients with suspiciously low accuracy
                if metrics["accuracy"] < 0.2:  # This threshold should be adjusted based on your use case
                    suspicious_indices.append(i)
                    logger.warning(f"Client {client_ids[i]} reported suspiciously low accuracy: {metrics['accuracy']}")
        
        # Calculate weighted average of losses with robustness
        weighted_losses = 0.0
        weighted_metrics = {}
        total_weight = sum(weights)
        
        # Normalize weights
        if total_weight == 0:
            weights = [1.0/len(results)] * len(results)
            total_weight = 1.0
        else:
            weights = [w/total_weight for w in weights]
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        all_weights = defaultdict(list)
        
        for i, (client, eval_res) in enumerate(results):
            # Skip suspicious clients
            if i in suspicious_indices:
                logger.info(f"Excluding suspicious client {client.cid} from evaluation aggregation")
                continue
                
            # Use client reputation to adjust weight
            reputation = self.client_reputations.get(client.cid, 0.0)
            adjusted_weight = weights[i] * max(0.1, reputation)  # Apply reputation but don't completely ignore
            
            weighted_losses += adjusted_weight * eval_res.loss
            
            # Collect metrics for robust aggregation
            for key, value in eval_res.metrics.items():
                try:
                    # Skip non-numeric metrics
                    if isinstance(value, (int, float)):
                        all_metrics[key].append(value)
                        all_weights[key].append(adjusted_weight)
                except Exception as e:
                    logger.warning(f"Error processing metric {key}: {e}")
        
        # Normalize weights for loss
        loss_weights_sum = sum(adjusted_weight for i, (_, _) in enumerate(results) 
                             if i not in suspicious_indices)
        if loss_weights_sum > 0:
            weighted_losses = weighted_losses / loss_weights_sum
        
        # Aggregate metrics with robust methods
        for key in all_metrics:
            values = np.array(all_metrics[key])
            metric_weights = all_weights[key]
            
            # Normalize weights
            total = sum(metric_weights)
            if total > 0:
                metric_weights = [w/total for w in metric_weights]
            else:
                metric_weights = [1.0/len(values)] * len(values)
            
            # Apply robust aggregation based on method
            if self.robust_method == "median":
                # Median is naturally robust to outliers
                weighted_metrics[key] = float(np.median(values))
            
            elif self.robust_method == "trimmed_mean":
                # Sort values
                sorted_idx = np.argsort(values)
                n_values = len(values)
                n_trim = int(n_values * self.trim_ratio)
                
                if n_trim > 0 and n_values > 2 * n_trim:
                    # Calculate trimmed mean (exclude lowest and highest values)
                    trimmed_values = values[sorted_idx[n_trim:-n_trim]]
                    trimmed_weights = [metric_weights[i] for i in sorted_idx[n_trim:-n_trim]]
                    
                    # Re-normalize weights
                    total = sum(trimmed_weights)
                    if total > 0:
                        trimmed_weights = [w/total for w in trimmed_weights]
                    else:
                        trimmed_weights = [1.0/len(trimmed_values)] * len(trimmed_values)
                    
                    # Weighted average of remaining values
                    weighted_metrics[key] = sum(v * w for v, w in zip(trimmed_values, trimmed_weights))
                else:
                    # Fall back to weighted average if not enough values
                    weighted_metrics[key] = sum(v * w for v, w in zip(values, metric_weights))
            
            else:  # Default to weighted average
                weighted_metrics[key] = sum(v * w for v, w in zip(values, metric_weights))
        
        # Add number of clients that participated
        weighted_metrics["num_clients"] = len(results)
        weighted_metrics["num_trusted_clients"] = len(results) - len(suspicious_indices)
        
        return weighted_losses, weighted_metrics


class ModelOutlierDetector:
    """
    Helper class for detecting outlier ML models in federated learning.
    Can be used by ByzantineRobustEnsembleAggregator as a more advanced outlier detection method.
    """
    
    def __init__(self, detection_method="pca", threshold=2.0):
        """
        Initialize the model outlier detector.
        
        Args:
            detection_method: Method to use for outlier detection ("pca", "knn", "z_score")
            threshold: Threshold for outlier detection
        """
        self.detection_method = detection_method
        self.threshold = threshold
    
    def extract_model_features(self, ensemble):
        """
        Extract numeric features from ensemble model for outlier detection.
        
        Args:
            ensemble: Ensemble state dictionary
            
        Returns:
            Flattened feature vector
        """
        # Extract model weights as feature vector
        feature_vector = []
        
        # Add ensemble weights
        feature_vector.extend(ensemble["weights"])
        
        # For each model, extract numeric parameters
        for state_dict in ensemble["model_state_dicts"]:
            for key, value in state_dict.items():
                # Skip non-numeric parameters
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    # If list is not too large, add all elements
                    if len(value) <= 100:
                        feature_vector.extend([float(x) for x in value])
                    else:
                        # If list is large, add summary statistics
                        values = np.array([float(x) for x in value])
                        feature_vector.append(float(np.mean(values)))
                        feature_vector.append(float(np.std(values)))
                        feature_vector.append(float(np.min(values)))
                        feature_vector.append(float(np.max(values)))
        
        return np.array(feature_vector)
    
    def detect_outliers(self, ensembles):
        """
        Detect outlier ensembles using statistical methods.
        
        Args:
            ensembles: List of ensemble state dictionaries
            
        Returns:
            List of indices of detected outliers
        """
        if len(ensembles) <= 2:
            # Not enough ensembles to detect outliers
            return []
        
        try:
            # Extract features for each ensemble
            feature_vectors = []
            for ensemble in ensembles:
                features = self.extract_model_features(ensemble)
                if features is not None and len(features) > 0:
                    feature_vectors.append(features)
            
            # If feature vectors have different lengths, use pairwise comparison
            if len(set(len(v) for v in feature_vectors)) != 1:
                return self._detect_outliers_pairwise(ensembles)
            
            # Convert to numpy array
            features = np.array(feature_vectors)
            
            # Detect outliers based on selected method
            if self.detection_method == "pca":
                return self._detect_outliers_pca(features)
            elif self.detection_method == "knn":
                return self._detect_outliers_knn(features)
            else:  # Default to z-score
                return self._detect_outliers_zscore(features)
                
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return []
    
    def _detect_outliers_zscore(self, features):
        """
        Detect outliers using z-score method.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            List of indices of detected outliers
        """
        # Calculate mean and std for each feature
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0) + 1e-10  # Avoid division by zero
        
        # Calculate z-scores for each client's features
        z_scores = np.abs((features - mean_features) / std_features)
        
        # Take mean z-score for each client across all features
        mean_z_scores = np.mean(z_scores, axis=1)
        
        # Flag outliers based on z-score threshold
        outliers = [i for i, z in enumerate(mean_z_scores) if z > self.threshold]
        
        return outliers
    
    def _detect_outliers_pca(self, features):
        """
        Detect outliers using PCA reconstruction error.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            List of indices of detected outliers
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Apply PCA
            n_components = min(3, min(scaled_features.shape) - 1)
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(scaled_features)
            
            # Reconstruct data
            reconstructed = pca.inverse_transform(pca_features)
            
            # Calculate reconstruction error
            errors = np.sum((scaled_features - reconstructed) ** 2, axis=1)
            
            # Determine threshold based on errors
            threshold = np.mean(errors) + self.threshold * np.std(errors)
            
            # Flag outliers based on reconstruction error
            outliers = [i for i, err in enumerate(errors) if err > threshold]
            
            return outliers
            
        except (ImportError, Exception) as e:
            logger.warning(f"Error using PCA for outlier detection: {e}. Falling back to z-score method.")
            return self._detect_outliers_zscore(features)
    
    def _detect_outliers_knn(self, features):
        """
        Detect outliers using K-nearest neighbors.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            List of indices of detected outliers
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Calculate distance to k nearest neighbors
            k = min(5, len(scaled_features) - 1)
            nn = NearestNeighbors(n_neighbors=k+1)  # +1 because the first neighbor is self
            nn.fit(scaled_features)
            distances, _ = nn.kneighbors(scaled_features)
            
            # Average distance to k nearest neighbors (exclude self)
            avg_distances = np.mean(distances[:, 1:], axis=1)
            
            # Determine threshold based on distances
            threshold = np.mean(avg_distances) + self.threshold * np.std(avg_distances)
            
            # Flag outliers based on average distance
            outliers = [i for i, dist in enumerate(avg_distances) if dist > threshold]
            
            return outliers
            
        except (ImportError, Exception) as e:
            logger.warning(f"Error using KNN for outlier detection: {e}. Falling back to z-score method.")
            return self._detect_outliers_zscore(features)
    
    def _detect_outliers_pairwise(self, ensembles):
        """
        Detect outliers using pairwise distances when feature vectors have different lengths.
        
        Args:
            ensembles: List of ensemble state dictionaries
            
        Returns:
            List of indices of detected outliers
        """
        n_ensembles = len(ensembles)
        distances = np.zeros((n_ensembles, n_ensembles))
        
        # Calculate pairwise distances between ensembles
        for i in range(n_ensembles):
            for j in range(i+1, n_ensembles):
                # Calculate distance based on model names
                names_i = set(ensembles[i]["model_names"])
                names_j = set(ensembles[j]["model_names"])
                name_similarity = len(names_i.intersection(names_j)) / max(1, len(names_i.union(names_j)))
                
                # Calculate distance based on ensemble weights
                weights_i = np.array(ensembles[i]["weights"])
                weights_j = np.array(ensembles[j]["weights"])
                
                # Align weights by model name
                aligned_weights_i = []
                aligned_weights_j = []
                
                common_names = sorted(names_i.intersection(names_j))
                for name in common_names:
                    idx_i = ensembles[i]["model_names"].index(name)
                    idx_j = ensembles[j]["model_names"].index(name)
                    aligned_weights_i.append(weights_i[idx_i])
                    aligned_weights_j.append(weights_j[idx_j])
                
                # If there are common models, calculate weight distance
                if aligned_weights_i:
                    weight_diff = np.mean(np.abs(np.array(aligned_weights_i) - np.array(aligned_weights_j)))
                else:
                    # If no common models, maximum difference
                    weight_diff = 1.0
                
                # Combined distance: weight difference scaled by name dissimilarity
                distance = weight_diff * (2.0 - name_similarity)
                distances[i, j] = distance
                distances[j, i] = distance
        
        # Calculate average distance to other clients
        avg_distances = np.sum(distances, axis=1) / (n_ensembles - 1)
        
        # Calculate z-scores of distances
        mean_dist = np.mean(avg_distances)
        std_dist = np.std(avg_distances) + 1e-10  # Avoid division by zero
        z_scores = (avg_distances - mean_dist) / std_dist
        
        # Flag outliers based on z-score threshold
        outliers = [i for i, z in enumerate(z_scores) if z > self.threshold]
        
        return outliers