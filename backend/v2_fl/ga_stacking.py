"""
GA-Stacking implementation for federated learning.
Provides genetic algorithm-based ensemble stacking optimization.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Any, Optional, Union
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("GA-Stacking")

class GAStacking:
    """Genetic Algorithm based Stacking Ensemble optimization."""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        model_names: List[str],
        population_size: int = 10,
        generations: int = 3,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize the GA-Stacking optimizer.
        
        Args:
            base_models: List of PyTorch models for base learners
            model_names: Names of the base models (for logging/tracking)
            population_size: Size of the GA population
            generations: Number of GA generations to run
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top individuals to keep without modification
            device: Device to use for computation ("cpu" or "cuda")
        """
        self.base_models = base_models
        self.model_names = model_names
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.device = torch.device(device)
        
        self.num_models = len(base_models)
        self.best_weights = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        # Ensure all models are on the correct device
        for model in self.base_models:
            model.to(self.device)
            model.eval()  # Set to evaluation mode
    
    def initialize_population(self) -> List[np.ndarray]:
        """
        Initialize the GA population with random weights.
        
        Returns:
            List of weight arrays for the population
        """
        population = []
        for _ in range(self.population_size):
            # Generate random weights and normalize
            weights = np.random.random(self.num_models)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            population.append(weights)
        
        return population
    
    def fitness_function(self, weights: np.ndarray, val_data: torch.utils.data.DataLoader) -> float:
        """
        Calculate fitness with strict separation of base and meta models.
        
        Args:
            weights: Array of weights for the ensemble
            val_data: Validation data loader
            
        Returns:
            Fitness score
        """
        correct = 0
        total = 0
        total_loss = 0.0
        
        # Separate weights for base models and meta-learners
        base_model_indices = []
        meta_model_indices = []
        
        for i, name in enumerate(self.model_names):
            if name == "meta_lr" or "meta" in name.lower():
                meta_model_indices.append(i)
            else:
                base_model_indices.append(i)
        
        base_weights = weights[base_model_indices]
        meta_weights = weights[meta_model_indices] if meta_model_indices else []
        
        # Normalize weights
        if len(base_weights) > 0:
            base_weights = base_weights / np.sum(base_weights)
        
        if len(meta_weights) > 0:
            meta_weights = meta_weights / np.sum(meta_weights)
        
        # Use eval context
        with torch.no_grad():
            for data, targets in val_data:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 1. Process base models
                base_outputs = []
                weighted_base_sum = None
                
                for i, idx in enumerate(base_model_indices):
                    try:
                        model = self.base_models[idx]
                        model.eval()
                        output = model(data)
                        
                        # Ensure tensor
                        if not isinstance(output, torch.Tensor):
                            output = torch.tensor(output, dtype=torch.float32, device=self.device)
                        
                        # Ensure proper shape
                        if len(output.shape) == 1:
                            output = output.unsqueeze(1)
                        
                        # Add to base outputs
                        base_outputs.append(output)
                        
                        # Apply weight
                        if i < len(base_weights):
                            weight = base_weights[i]
                            weighted_output = output * weight
                            
                            # Add to weighted sum
                            if weighted_base_sum is None:
                                weighted_base_sum = weighted_output
                            else:
                                weighted_base_sum += weighted_output
                    except Exception as e:
                        print(f"Error in base model forward pass during fitness evaluation: {e}")
                        # Skip this model
                
                # 2. Process meta-learners if we have base outputs
                weighted_meta_sum = None
                
                if len(base_outputs) > 0 and len(meta_model_indices) > 0:
                    try:
                        # Create meta-learner input
                        meta_input = torch.cat(base_outputs, dim=1)
                        
                        # Process through meta-learners
                        for i, idx in enumerate(meta_model_indices):
                            try:
                                model = self.base_models[idx]
                                model.eval()
                                output = model(meta_input)  # Use base model predictions!
                                
                                # Ensure tensor
                                if not isinstance(output, torch.Tensor):
                                    output = torch.tensor(output, dtype=torch.float32, device=self.device)
                                
                                # Ensure proper shape
                                if len(output.shape) == 1:
                                    output = output.unsqueeze(1)
                                
                                # Apply weight
                                if i < len(meta_weights):
                                    weight = meta_weights[i]
                                    weighted_output = output * weight
                                    
                                    # Add to weighted sum
                                    if weighted_meta_sum is None:
                                        weighted_meta_sum = weighted_output
                                    else:
                                        weighted_meta_sum += weighted_output
                            except Exception as e:
                                print(f"Error in meta-learner forward pass during fitness evaluation: {e}")
                                # Skip this model
                    except Exception as e:
                        print(f"Error preparing meta-learner input during fitness evaluation: {e}")
                        # Skip meta-learners
                
                # 3. Combine outputs
                final_output = None
                
                if weighted_base_sum is not None:
                    final_output = weighted_base_sum
                
                if weighted_meta_sum is not None:
                    if final_output is None:
                        final_output = weighted_meta_sum
                    else:
                        final_output += weighted_meta_sum
                
                # Skip this batch if no valid output
                if final_output is None:
                    continue
                
                # 4. Calculate metrics
                # For regression task (fraud detection)
                if targets.dim() == 1 or targets.size(1) == 1:
                    # Calculate MSE
                    mse = torch.mean((final_output - targets) ** 2).item()
                    total_loss += mse * targets.size(0)
                    # Count correct predictions (within threshold)
                    threshold = 0.5
                    correct += torch.sum((torch.abs(final_output - targets) < threshold)).item()
                # For classification
                else:
                    _, predicted = torch.max(final_output, 1)
                    _, target_classes = torch.max(targets, 1)
                    correct += (predicted == target_classes).sum().item()
                    
                total += targets.size(0)
        
        # Calculate final fitness
        if total == 0:
            return 0.0
            
        accuracy = correct / total
        avg_loss = total_loss / total if total > 0 else float('inf')
        
        # Include diversity in fitness
        weight_diversity = np.std(weights)
        
        # Combined fitness (higher is better)
        combined_fitness = accuracy + (weight_diversity * 0.2)
        
        return combined_fitness
    
    def select_parents(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """
        Select parents using tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for the population
            
        Returns:
            Selected parents
        """
        parents = []
        for _ in range(self.population_size):
            # Tournament selection with size 3
            tournament_indices = random.sample(range(self.population_size), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent weights
            parent2: Second parent weights
            
        Returns:
            Two offspring weight arrays
        """
        if random.random() < self.crossover_rate:
            # Single point crossover
            point = random.randint(1, self.num_models - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            
            # Normalize weights to sum to 1
            child1 = child1 / np.sum(child1)
            child2 = child2 / np.sum(child2)
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Weight array to mutate
            
        Returns:
            Mutated weight array
        """
        for i in range(self.num_models):
            if random.random() < self.mutation_rate:
                # Add Gaussian noise
                individual[i] += np.random.normal(0, 0.1)
                
                # Ensure no negative weights
                individual[i] = max(0, individual[i])
        
        # Re-normalize weights to sum to 1
        individual = individual / np.sum(individual)
        
        return individual
    
    def evolve_population(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """
        Evolve the population to the next generation.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for the population
            
        Returns:
            New population
        """
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        # Keep elite individuals
        new_population = sorted_population[:self.elite_size]
        
        # Selection
        parents = self.select_parents(population, fitness_scores)
        
        # Crossover and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population
    
    def optimize(
        self, 
        train_data: torch.utils.data.DataLoader, 
        val_data: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Run the GA optimization process.
        
        Args:
            train_data: Training data loader
            val_data: Validation data loader
            
        Returns:
            Best weights found for the ensemble
        """
        # Initialize population
        population = self.initialize_population()
        
        # Run GA for specified number of generations
        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitness_scores = []
            for individual in population:
                fitness = self.fitness_function(individual, val_data)
                fitness_scores.append(fitness)
            
            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[max_fitness_idx]
            current_best_weights = population[max_fitness_idx]
            
            # Update global best if needed
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_weights = current_best_weights.copy()
            
            # Add to history
            self.fitness_history.append({
                "generation": generation,
                "best_fitness": current_best_fitness,
                "avg_fitness": np.mean(fitness_scores),
                "best_weights": current_best_weights.tolist()
            })
            
            # Log progress
            logger.info(f"Generation {generation+1}/{self.generations}, "
                       f"Best Fitness: {current_best_fitness:.4f}, "
                       f"Avg Fitness: {np.mean(fitness_scores):.4f}")
            
            # Evolve population
            if generation < self.generations - 1:
                population = self.evolve_population(population, fitness_scores)
        
        # Log final weights
        weight_str = ", ".join([f"{self.model_names[i]}: {self.best_weights[i]:.4f}" 
                               for i in range(self.num_models)])
        logger.info(f"Optimization complete. Best weights: {weight_str}")
        
        return self.best_weights
    
    def get_ensemble_model(self) -> 'EnsembleModel':
        """
        Create an ensemble model with the optimized weights.
        
        Returns:
            Weighted ensemble model
        """
        if self.best_weights is None:
            raise ValueError("Must run optimize() before getting the ensemble model")
        
        return EnsembleModel(
            models=self.base_models,
            weights=self.best_weights,
            model_names=self.model_names,
            device=self.device
        )


class EnsembleModel(nn.Module):
    """Weighted ensemble model that combines multiple base models."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: np.ndarray,
        model_names: List[str],
        device: str = "cpu"
    ):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of base models
            weights: Array of weights for the models
            model_names: Names of the base models
            device: Device to use for computation
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights, device=device)
        self.model_names = model_names
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble with strict separation between base models and meta-learners.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted output from the ensemble
        """
        # Handle non-tensor inputs and ensure consistent dtype
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            # Ensure consistent dtype to avoid Float/Double issues
            x = x.to(dtype=torch.float32, device=self.device)
        
        # 1. First identify base models and meta-learners
        base_models = []
        base_weights = []
        base_model_indices = []
        meta_models = []
        meta_weights = []
        meta_model_indices = []
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            if name == "meta_lr" or "meta" in name.lower():
                meta_models.append(model)
                meta_weights.append(self.weights[i])
                meta_model_indices.append(i)
            else:
                base_models.append(model)
                base_weights.append(self.weights[i])
                base_model_indices.append(i)
        
        # 2. CRITICAL SECTION: PROCESS BASE MODELS FIRST
        base_model_outputs = []
        weighted_base_outputs = None
        
        for i, model in enumerate(base_models):
            try:
                # Process input through base model
                with torch.no_grad():
                    model.eval()
                    base_output = model(x)
                    
                    # Ensure output is proper tensor and shape
                    if not isinstance(base_output, torch.Tensor):
                        base_output = torch.tensor(base_output, dtype=torch.float32, device=self.device)
                    
                    if len(base_output.shape) == 1:
                        base_output = base_output.unsqueeze(1)
                    
                    # Store for meta-learner input
                    base_model_outputs.append(base_output)
                    
                    # Apply weight and add to weighted sum
                    weight = base_weights[i]
                    weighted_output = base_output * weight
                    
                    if weighted_base_outputs is None:
                        weighted_base_outputs = weighted_output
                    else:
                        weighted_base_outputs += weighted_output
            except Exception as e:
                logger.error(f"Error in base model forward pass: {e}")
                # Skip this model
        
        # If we have no valid base model outputs, return zeros
        if not base_model_outputs:
            return torch.zeros((x.size(0), 1), dtype=torch.float32, device=self.device)
        
        # 3. NOW PROCESS META-LEARNERS WITH BASE MODEL OUTPUTS
        weighted_meta_outputs = None
        
        if meta_models and len(base_model_outputs) > 0:
            try:
                # Concatenate base model outputs for meta-learner input
                # This creates a tensor of shape [batch_size, num_base_models]
                meta_learner_input = torch.cat(base_model_outputs, dim=1)
                
                # Explicitly log the dimensions to verify
                logger.debug(f"Meta-learner input shape: {meta_learner_input.shape} - correct format for meta-learners")
                
                # Process through each meta-learner
                for i, model in enumerate(meta_models):
                    try:
                        # Forward pass with base model outputs ONLY
                        with torch.no_grad():
                            model.eval()
                            meta_output = model(meta_learner_input)
                            
                            # Ensure proper shape
                            if not isinstance(meta_output, torch.Tensor):
                                meta_output = torch.tensor(meta_output, dtype=torch.float32, device=self.device)
                            
                            if len(meta_output.shape) == 1:
                                meta_output = meta_output.unsqueeze(1)
                            
                            # Apply weight
                            weight = meta_weights[i]
                            weighted_output = meta_output * weight
                            
                            # Add to weighted sum
                            if weighted_meta_outputs is None:
                                weighted_meta_outputs = weighted_output
                            else:
                                weighted_meta_outputs += weighted_output
                    except Exception as e:
                        logger.error(f"Error in meta-learner forward pass: {e}")
                        # Skip this meta-learner
            except Exception as e:
                logger.error(f"Error preparing meta-learner input: {e}")
                # Skip meta-learners entirely
        
        # 4. COMBINE OUTPUTS
        final_output = None
        
        # Add base model outputs if available
        if weighted_base_outputs is not None:
            final_output = weighted_base_outputs
        
        # Add meta-learner outputs if available
        if weighted_meta_outputs is not None:
            if final_output is None:
                final_output = weighted_meta_outputs
            else:
                final_output += weighted_meta_outputs
        
        # Return zeros if no valid outputs
        if final_output is None:
            return torch.zeros((x.size(0), 1), dtype=torch.float32, device=self.device)
        
        return final_output
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get the model weights as a dictionary.
        
        Returns:
            Dictionary mapping model names to weights
        """
        return {name: float(weight) for name, weight in zip(self.model_names, self.weights)}