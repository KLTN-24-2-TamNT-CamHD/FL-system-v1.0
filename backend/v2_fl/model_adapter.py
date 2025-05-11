import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback

# Import scikit-learn models for direct wrapping
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    sklearn_available = True
except ImportError:
    sklearn_available = False

# Import boosting libraries if available
try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgboost_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

try:
    import catboost as cb
    catboost_available = True
except ImportError:
    catboost_available = False

logger = logging.getLogger(__name__)

class ModelAdapter:
    """
    Adapter class that manages both PyTorch and scikit-learn models,
    providing a consistent interface for both types.
    
    This solves compatibility issues between different model types.
    """
    
    def __init__(self, model_type: str, input_dim: int, output_dim: int = 1, device: str = "cpu", **kwargs):
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.kwargs = kwargs
        self.meta_learner = None

        if model_type.lower() == "meta_lr":
            self.meta_learner = self

        # Create the appropriate model
        self.model = self._create_model()
        self.is_torch_model = isinstance(self.model, nn.Module)
        
        logger.info(f"Created {model_type} model with input_dim={input_dim}, output_dim={output_dim}")
        
    def _create_model(self) -> Any:
        """Create the appropriate model based on model_type."""
        try:
            if self.model_type == "lr":
                return self._create_logistic_regression()
            elif self.model_type == "svc":
                return self._create_svc()
            elif self.model_type == "rf":
                return self._create_random_forest()
            elif self.model_type == "knn":
                return self._create_knn()
            elif self.model_type == "xgb":
                return self._create_xgboost()
            elif self.model_type == "lgbm":
                return self._create_lightgbm()
            elif self.model_type == "catboost":
                return self._create_catboost()
            elif self.model_type == "meta_lr":
                return self._create_meta_learner()
            else:
                logger.warning(f"Unknown model type: {self.model_type}, using simple model")
                return SimpleModel(self.input_dim, self.output_dim).to(self.device)
        except Exception as e:
            logger.error(f"Error creating model {self.model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to simple model
            return SimpleModel(self.input_dim, self.output_dim).to(self.device)
            
    def _create_logistic_regression(self) -> Any:
        """Create logistic regression model."""
        if sklearn_available:
            # Use sklearn LogisticRegression for better performance and compatibility
            model = LogisticRegression(
                penalty=self.kwargs.get("penalty", "l2"),
                C=self.kwargs.get("C", 1.0),
                class_weight=self.kwargs.get("class_weight", "balanced"),
                solver=self.kwargs.get("solver", "liblinear"),
                max_iter=self.kwargs.get("max_iter", 1000),
                random_state=self.kwargs.get("random_state", 42),
                n_jobs=-1
            )
            # Initialize with provided coefficients if available
            if "coef" in self.kwargs and "intercept" in self.kwargs:
                try:
                    model.coef_ = np.array(self.kwargs["coef"])
                    model.intercept_ = np.array(self.kwargs["intercept"])
                    model.classes_ = np.array([0, 1])  # Binary classification
                    # Mark as fitted
                    model._fitted = True
                except Exception as e:
                    logger.warning(f"Error setting LogisticRegression parameters: {str(e)}")
            return model
        else:
            # Fall back to PyTorch implementation
            return LogisticRegressionModel(self.input_dim, self.output_dim).to(self.device)
            
    def _create_svc(self) -> Any:
        """Create support vector classifier."""
        if sklearn_available:
            # Use sklearn SVC for better performance and compatibility
            model = SVC(
                kernel=self.kwargs.get("kernel", "rbf"),
                C=self.kwargs.get("C", 10.0),
                gamma=self.kwargs.get("gamma", "scale"),
                probability=True,
                class_weight=self.kwargs.get("class_weight", "balanced"),
                random_state=self.kwargs.get("random_state", 42)
            )
            # Try to initialize with provided parameters
            try:
                if all(key in self.kwargs for key in ["dual_coef", "support_vectors", "intercept"]):
                    model.dual_coef_ = np.array(self.kwargs["dual_coef"])
                    model.support_vectors_ = np.array(self.kwargs["support_vectors"])
                    model.intercept_ = np.array(self.kwargs["intercept"])
                    model.classes_ = np.array([0, 1])  # Binary classification
                    # Set necessary attributes to mark as fitted
                    model._sparse = False
                    model.shape_fit_ = (self.input_dim,)
                    model._n_support = np.array([1, 1])  # One support vector per class
                    model.fit_status_ = 0
                    # Mark as fitted
                    model._fitted = True
            except Exception as e:
                logger.warning(f"Error setting SVC parameters: {str(e)}")
            return model
        else:
            # Fall back to PyTorch implementation
            return SVCModel(self.input_dim, self.output_dim).to(self.device)
            
    def _create_random_forest(self) -> Any:
        """Create random forest classifier."""
        if sklearn_available:
            # Use sklearn RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                criterion=self.kwargs.get("criterion", "gini"),
                max_depth=self.kwargs.get("max_depth", 15),
                min_samples_split=self.kwargs.get("min_samples_split", 2),
                min_samples_leaf=self.kwargs.get("min_samples_leaf", 2),
                max_features=self.kwargs.get("max_features", "sqrt"),
                bootstrap=self.kwargs.get("bootstrap", True),
                class_weight=self.kwargs.get("class_weight", "balanced_subsample"),
                random_state=self.kwargs.get("random_state", 42),
                n_jobs=-1
            )
            # Initialize feature importances if available
            if "feature_importances" in self.kwargs:
                try:
                    # We can't fully initialize a RandomForest without training
                    # but we can set some attributes to avoid initial errors
                    model.n_features_in_ = self.input_dim
                    model.classes_ = np.array([0, 1])  # Binary classification
                    model.n_classes_ = 2
                    model.feature_importances_ = np.array(self.kwargs["feature_importances"])
                except Exception as e:
                    logger.warning(f"Error setting RandomForest parameters: {str(e)}")
            return model
        else:
            # Fall back to PyTorch implementation
            return RandomForestModel(self.input_dim, self.output_dim).to(self.device)
            
    def _create_knn(self) -> Any:
        """Create k-nearest neighbors classifier."""
        if sklearn_available:
            # Use sklearn KNeighborsClassifier
            model = KNeighborsClassifier(
                n_neighbors=self.kwargs.get("n_neighbors", 5),
                weights=self.kwargs.get("weights", "distance"),
                algorithm=self.kwargs.get("algorithm", "auto"),
                leaf_size=self.kwargs.get("leaf_size", 30),
                p=self.kwargs.get("p", 2),
                metric=self.kwargs.get("metric", "minkowski"),
                metric_params=self.kwargs.get("metric_params", None),
                n_jobs=-1
            )
            # We can't fully initialize KNN without data
            # but we can set some attributes to avoid initial errors
            try:
                model.n_features_in_ = self.input_dim
                model.classes_ = np.array([0, 1])  # Binary classification
                # Create minimal synthetic data for initialization
                X = np.random.randn(10, self.input_dim)
                y = np.random.randint(0, 2, 10)
                model.fit(X, y)
            except Exception as e:
                logger.warning(f"Error initializing KNN: {str(e)}")
            return model
        else:
            # Fall back to PyTorch implementation
            return KNNModel(self.input_dim, self.output_dim).to(self.device)
            
    def _create_xgboost(self) -> Any:
        """Create XGBoost classifier."""
        if xgboost_available:
            # Use XGBoost
            params = {
                "n_estimators": self.kwargs.get("n_estimators", 100),
                "max_depth": self.kwargs.get("max_depth", 6),
                "learning_rate": self.kwargs.get("learning_rate", 0.1),
                "subsample": self.kwargs.get("subsample", 0.8),
                "colsample_bytree": self.kwargs.get("colsample_bytree", 0.8),
                "objective": self.kwargs.get("objective", "binary:logistic"),
                "scale_pos_weight": self.kwargs.get("scale_pos_weight", 10.0),
                "random_state": self.kwargs.get("random_state", 42),
                "n_jobs": -1
            }
            model = xgb.XGBClassifier(**params)
            # Initialize feature importances if available
            if "feature_importances" in self.kwargs:
                try:
                    # Create a minimal model to set feature importances
                    model.n_features_in_ = self.input_dim
                    model.classes_ = np.array([0, 1])  # Binary classification
                    # Create minimal booster
                    model._Booster = xgb.Booster()
                    model._Booster.feature_names = [f'f{i}' for i in range(self.input_dim)]
                except Exception as e:
                    logger.warning(f"Error setting XGBoost parameters: {str(e)}")
            return model
        else:
            # Fall back to PyTorch implementation
            return XGBoostModel(self.input_dim, self.output_dim).to(self.device)
            
    def _create_lightgbm(self) -> Any:
        """Create LightGBM classifier."""
        if lightgbm_available:
            # Use LightGBM
            params = {
                "n_estimators": self.kwargs.get("n_estimators", 100),
                "max_depth": self.kwargs.get("max_depth", 8),
                "learning_rate": self.kwargs.get("learning_rate", 0.05),
                "subsample": self.kwargs.get("subsample", 0.8),
                "colsample_bytree": self.kwargs.get("colsample_bytree", 0.8),
                "objective": self.kwargs.get("objective", "binary"),
                "class_weight": self.kwargs.get("class_weight", "balanced"),
                "boosting_type": self.kwargs.get("boosting_type", "gbdt"),
                "importance_type": self.kwargs.get("importance_type", "gain"),
                "random_state": self.kwargs.get("random_state", 42),
                "n_jobs": -1
            }
            model = lgb.LGBMClassifier(**params)
            # Initialize feature importances if available
            if "feature_importances" in self.kwargs:
                try:
                    model.n_features_in_ = self.input_dim
                    model.classes_ = np.array([0, 1])  # Binary classification
                    # We can't fully initialize LightGBM without training
                except Exception as e:
                    logger.warning(f"Error setting LightGBM parameters: {str(e)}")
            return model
        else:
            # Fall back to PyTorch implementation
            return LightGBMModel(self.input_dim, self.output_dim).to(self.device)
    
    def _create_catboost(self) -> Any:
        """Create CatBoost classifier."""
        if catboost_available:
            # Use CatBoost
            params = {
                "iterations": self.kwargs.get("iterations", 100),
                "depth": self.kwargs.get("depth", 6),
                "learning_rate": self.kwargs.get("learning_rate", 0.1),
                "loss_function": self.kwargs.get("loss_function", "Logloss"),
                "verbose": False,
                "random_seed": self.kwargs.get("random_state", 42),
                "thread_count": -1
            }
            # Handle class weights differently for CatBoost
            if "class_weights" in self.kwargs:
                params["class_weights"] = self.kwargs["class_weights"]
                
            model = cb.CatBoostClassifier(**params)
            # Initialize feature importances if available
            if "feature_importances" in self.kwargs:
                try:
                    model._feature_count = self.input_dim
                    model.classes_ = np.array([0, 1])  # Binary classification
                except Exception as e:
                    logger.warning(f"Error setting CatBoost parameters: {str(e)}")
            return model
        else:
            # Fall back to PyTorch implementation
            return CatBoostModel(self.input_dim, self.output_dim).to(self.device)
            
    def _create_meta_learner(self) -> Any:
        """Create meta-learner model."""
        # For meta-learner, the input dimension is the number of base models
        # Always use PyTorch implementation for meta-learner
        return MetaLearnerModel(self.input_dim, self.output_dim).to(self.device)
    
    def fit(self, X, y):
        """Fit the model to the data."""
        try:
            if self.is_torch_model:
                # For PyTorch models, we need to implement training
                # This is just a placeholder
                if hasattr(self.model, 'train_model'):
                    self.model.train_model(X, y)
                else:
                    logger.warning(f"PyTorch model {self.model_type} does not have train_model method")
            else:
                # For sklearn/other models, use their fit method
                self.model.fit(X, y)
            return self
        except Exception as e:
            logger.error(f"Error fitting {self.model_type} model: {str(e)}")
            logger.error(traceback.format_exc())
            return self
    
    def predict(self, X):
        """Get predictions from the model."""
        try:
            # Convert data if needed
            X_np = self._convert_to_numpy(X)
            
            if self.is_torch_model:
                # For PyTorch models
                X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    outputs = self.model(X_tensor)
                    predictions = (outputs > 0.5).float().cpu().numpy()
                return predictions
            else:
                # For sklearn models
                if hasattr(self.model, 'predict'):
                    return self.model.predict(X_np)
                else:
                    logger.error(f"Model {self.model_type} does not have predict method")
                    return np.zeros((X_np.shape[0], 1))
        except Exception as e:
            logger.error(f"Error in prediction for {self.model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            return np.zeros((len(X), 1))
            
    def predict_proba(self, X):
        """Get probability predictions from the model."""
        try:
            # Convert data if needed
            X_np = self._convert_to_numpy(X)
            
            if self.is_torch_model:
                # For PyTorch models
                X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    outputs = self.model(X_tensor)
                    probs = outputs.cpu().numpy()
                    # Convert to [1-p, p] format for compatibility
                    return np.hstack([1 - probs, probs])
            else:
                # For sklearn models
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X_np)
                elif hasattr(self.model, 'predict'):
                    # Fallback to binary predictions
                    preds = self.model.predict(X_np)
                    # Convert to probability-like format [1-p, p]
                    return np.hstack([1 - preds.reshape(-1, 1), preds.reshape(-1, 1)])
                else:
                    logger.error(f"Model {self.model_type} does not have predict_proba method")
                    return np.zeros((X_np.shape[0], 2))
        except Exception as e:
            logger.error(f"Error in probability prediction for {self.model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            return np.zeros((len(X), 2))
    
    def forward(self, X):
        """Forward pass for both PyTorch and sklearn models."""
        try:
            # Check for meta_learner access for backwards compatibility
            if hasattr(X, 'meta_learner') and X.meta_learner is self:
                # This is being called in a meta-learner context
                # X is actually supposed to be the stacked outputs from base models
                # Just pass it through
                return self.model(X)
                
            # If X is a tensor, convert to numpy for sklearn models
            if isinstance(X, torch.Tensor):
                if self.is_torch_model:
                    # For PyTorch models, use directly
                    return self.model(X)
                else:
                    # Convert to numpy for sklearn models
                    X_np = X.cpu().numpy()
                    # Get probabilities and return the positive class prob
                    probs = self.predict_proba(X_np)
                    # Convert back to tensor
                    return torch.tensor(probs[:, 1].reshape(-1, 1), dtype=torch.float32, device=self.device)
            else:
                # If X is numpy, convert to tensor for PyTorch models
                if self.is_torch_model:
                    X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                    return self.model(X_tensor)
                else:
                    # Get probabilities and return the positive class prob
                    probs = self.predict_proba(X)
                    # Convert to tensor
                    return torch.tensor(probs[:, 1].reshape(-1, 1), dtype=torch.float32, device=self.device)
        except Exception as e:
            logger.error(f"Error in forward pass for {self.model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            if isinstance(X, torch.Tensor):
                return torch.zeros((X.shape[0], 1), device=self.device)
            else:
                return torch.zeros((X.shape[0], 1), device=self.device)
    
    def _convert_to_numpy(self, X):
        """Convert input data to numpy if it's a tensor."""
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return X
        
    def __call__(self, X):
        """Make the adapter callable like a model."""
        return self.forward(X)
        
    def to(self, device):
        """Move PyTorch model to device."""
        self.device = device
        if self.is_torch_model:
            self.model = self.model.to(device)
        return self
        
    def eval(self):
        """Set PyTorch model to evaluation mode."""
        if self.is_torch_model and hasattr(self.model, 'eval'):
            self.model.eval()
        return self
        
    def train(self):
        """Set PyTorch model to training mode."""
        if self.is_torch_model and hasattr(self.model, 'train'):
            if callable(self.model.train):
                self.model.train()
        return self
        
    def get_params(self, deep=True):
        """Get model parameters (for sklearn compatibility)."""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params(deep=deep)
        return {}
        
    def set_params(self, **params):
        """Set model parameters (for sklearn compatibility)."""
        if hasattr(self.model, 'set_params'):
            return self.model.set_params(**params)
        return self
        
    def set_parameters(self, params):
        """Set model parameters from dictionary."""
        if hasattr(self.model, 'set_parameters'):
            self.model.set_parameters(params)
        elif hasattr(self.model, 'set_params'):
            # Extract sklearn compatible params
            sklearn_params = {}
            for key, value in params.items():
                # Filter out the keys that are not model parameters
                if key not in [
                    'model_type', 'estimator', 'input_dim', 'output_dim', 
                    'coef', 'intercept', 'dual_coef', 'support_vectors',
                    'feature_importances'
                ]:
                    sklearn_params[key] = value
            if sklearn_params:
                self.model.set_params(**sklearn_params)
                    
        # Handle specific parameters for different model types
        try:
            if self.model_type == "lr" and not self.is_torch_model:
                if "coef" in params and "intercept" in params:
                    # Handle different formats of coefficients
                    coef_data = params["coef"]
                    if isinstance(coef_data, list):
                        if len(coef_data) == 1 and len(coef_data[0]) == self.input_dim:
                            # Single row of coefficients
                            coef_array = np.array(coef_data)
                        elif len(coef_data) == self.input_dim:
                            # Flat list of coefficients
                            coef_array = np.array([coef_data])
                        else:
                            logger.warning(f"Invalid coefficient shape: {np.array(coef_data).shape}, expected (1, {self.input_dim})")
                            coef_array = np.zeros((1, self.input_dim))
                    else:
                        coef_array = np.array(coef_data)
                        
                    # Set coefficients and intercept
                    if hasattr(self.model, 'coef_'):
                        self.model.coef_ = coef_array
                    if hasattr(self.model, 'intercept_'):
                        self.model.intercept_ = np.array(params["intercept"])
                    
                    # Set classes for classification
                    if hasattr(self.model, 'classes_'):
                        self.model.classes_ = np.array([0, 1])  # Binary classification
                    
                    # Mark as fitted
                    if hasattr(self.model, '_fitted'):
                        self.model._fitted = True
            elif self.model_type == "meta_lr" and self.is_torch_model:
                if "coef" in params and "intercept" in params:
                    if hasattr(self.model, 'linear'):
                        coef_data = params["coef"]
                        intercept_data = params["intercept"]
                        
                        # Handle different formats
                        if isinstance(coef_data, list) and len(coef_data) == 1:
                            coef_tensor = torch.tensor(coef_data, dtype=torch.float32, device=self.device)
                        else:
                            coef_tensor = torch.tensor([coef_data], dtype=torch.float32, device=self.device)
                            
                        # Set weights and bias
                        self.model.linear.weight.data = coef_tensor
                        self.model.linear.bias.data = torch.tensor(intercept_data, dtype=torch.float32, device=self.device)
        except Exception as e:
            logger.warning(f"Error setting specific parameters for {self.model_type}: {str(e)}")
        
        return self

# PyTorch model implementations (same as in previous code)
class SimpleModel(nn.Module):
    """Simple neural network model for binary classification."""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class LogisticRegressionModel(nn.Module):
    """PyTorch implementation of logistic regression."""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LogisticRegressionModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
        
    def set_parameters(self, params: Dict[str, Any]):
        """Set model parameters from configuration."""
        try:
            if "coef" in params and "intercept" in params:
                coef = torch.tensor(params["coef"], dtype=torch.float32)
                intercept = torch.tensor(params["intercept"], dtype=torch.float32)
                
                # Handle dimension issues
                if coef.dim() == 2 and coef.size(0) == 1:
                    coef = coef.transpose(0, 1)
                
                if intercept.dim() == 1 and intercept.size(0) == 1:
                    intercept = intercept.squeeze(0)
                
                # Set parameters
                self.linear.weight.data = coef
                self.linear.bias.data = intercept
        except Exception as e:
            logger.error(f"Error setting parameters: {str(e)}")

class SVCModel(nn.Module):
    """PyTorch implementation of SVC."""
    def __init__(self, input_dim: int, output_dim: int = 1, kernel="rbf"):
        super(SVCModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel = kernel
        
        # For linear kernel or initial approximation
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class RandomForestModel(nn.Module):
    """PyTorch implementation of Random Forest."""
    def __init__(self, input_dim: int, output_dim: int = 1, n_estimators: int = 100):
        super(RandomForestModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        
        # Feature importance weights
        self.feature_weights = nn.Parameter(torch.ones(input_dim) / input_dim)
        
        # Output layer
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Apply feature importance
        weighted_features = x * self.feature_weights
        
        # Simple output layer
        out = self.linear(weighted_features)
        return self.sigmoid(out)

class KNNModel(nn.Module):
    """PyTorch implementation of KNN."""
    def __init__(self, input_dim: int, output_dim: int = 1, n_neighbors: int = 5):
        super(KNNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neighbors = n_neighbors
        
        # For simplified implementation
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

class GradientBoostingModel(nn.Module):
    """Base class for gradient boosting models."""
    def __init__(self, input_dim: int, output_dim: int = 1, n_estimators: int = 100):
        super(GradientBoostingModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        
        # Feature importance weights
        self.feature_weights = nn.Parameter(torch.ones(input_dim) / input_dim)
        
        # Multiple layers to simulate boosting
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply feature importance
        weighted_features = x * self.feature_weights
        
        # Pass through layers
        return self.layers(weighted_features)

class XGBoostModel(GradientBoostingModel):
    """PyTorch implementation of XGBoost."""
    def __init__(self, input_dim: int, output_dim: int = 1, n_estimators: int = 100):
        super(XGBoostModel, self).__init__(input_dim, output_dim, n_estimators)
        self.model_type = "xgb"

class LightGBMModel(GradientBoostingModel):
    """PyTorch implementation of LightGBM."""
    def __init__(self, input_dim: int, output_dim: int = 1, n_estimators: int = 100):
        super(LightGBMModel, self).__init__(input_dim, output_dim, n_estimators)
        self.model_type = "lgbm"

class CatBoostModel(GradientBoostingModel):
    """PyTorch implementation of CatBoost."""
    def __init__(self, input_dim: int, output_dim: int = 1, n_estimators: int = 100):
        super(CatBoostModel, self).__init__(input_dim, output_dim, n_estimators)
        self.model_type = "catboost"

class MetaLearnerModel(nn.Module):
    """Meta-learner model for stacking the base models."""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(MetaLearnerModel, self).__init__()
        self.input_dim = input_dim  # Number of base models
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize with equal weights
        nn.init.constant_(self.linear.weight, 1.0 / input_dim)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        # Check if input dimension matches expected
        if isinstance(x, torch.Tensor) and x.size(1) != self.input_dim:
            logger.error(f"Input has {x.size(1)} features, but model expects {self.input_dim}")
            if x.size(1) > self.input_dim:
                # Too many features, truncate
                x = x[:, :self.input_dim]
            else:
                # Too few features, pad with zeros
                padding = torch.zeros(x.size(0), self.input_dim - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
                
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
        
    def set_parameters(self, params: Dict[str, Any]):
        """Set model parameters from configuration."""
        try:
            # Set coefficients
            if "coef" in params:
                coef = torch.tensor(params["coef"], dtype=torch.float32)
                
                # Handle dimension issues
                if coef.dim() == 2 and coef.size(0) == 1:
                    # Reshape to match linear layer expected shape [output_dim, input_dim]
                    coef = coef.transpose(0, 1)
                
                # Ensure correct shape
                if coef.size(1) != self.input_dim:
                    logger.warning(f"Meta-learner coef dimension mismatch: got {coef.size(1)}, expected {self.input_dim}")
                    # Adjust by padding or truncating
                    if coef.size(1) < self.input_dim:
                        # Pad with zeros
                        padding = torch.zeros(coef.size(0), self.input_dim - coef.size(1))
                        coef = torch.cat([coef, padding], dim=1)
                    else:
                        # Truncate
                        coef = coef[:, :self.input_dim]
                
                self.linear.weight.data = coef
                
            # Set intercept
            if "intercept" in params:
                intercept = torch.tensor(params["intercept"], dtype=torch.float32)
                if intercept.dim() == 1:
                    self.linear.bias.data = intercept
                    
            logger.info(f"Meta-learner parameters set: weight shape {self.linear.weight.shape}")
        except Exception as e:
            logger.error(f"Error setting meta-learner parameters: {str(e)}")
            logger.error(traceback.format_exc())

# Enhanced Ensemble Model that works with the Adapter
class AdaptiveEnsembleModel(nn.Module):
    """Ensemble model that works with both PyTorch and sklearn models through the adapter."""
    def __init__(
        self, 
        models: List[Any], 
        weights: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
        model_names: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        super(AdaptiveEnsembleModel, self).__init__()
        self.models = models  # List of ModelAdapter or nn.Module
        self.device = device
        
        # Initialize weights if not provided
        if weights is None:
            weights = torch.ones(len(models), device=device) / len(models)
        else:
            # Convert weights to tensor if they're not already
            if isinstance(weights, (np.ndarray, list)):
                weights = torch.tensor(weights, dtype=torch.float32, device=device)
            else:
                weights = weights.to(device)
            
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        # Register weights as parameter
        self.weights = nn.Parameter(weights)
        
        # Store model names
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(models))]
        self.model_names = model_names[:len(models)]  # Ensure matching length
        
        # Meta-learner for stacking if available
        self.meta_learner = None
        for i, model in enumerate(models):
            model_type = getattr(model, 'model_type', '')
            if model_type == "meta_lr" or 'meta_lr' in self.model_names[i]:
                self.meta_learner = model
                break
                
        logger.info(f"Created ensemble with {len(models)} models and {'with' if self.meta_learner else 'without'} meta-learner")
        
    def forward(self, x):
        """Forward pass through ensemble."""
        # If we have a meta-learner, use it for stacking
        if self.meta_learner is not None:
            try:
                # Get predictions from base models
                base_outputs = []
                
                for i, model in enumerate(self.models):
                    # Skip meta-learner itself
                    if model is self.meta_learner:
                        continue
                        
                    # Get output from base model
                    try:
                        with torch.no_grad():
                            # Direct forward pass without checking for meta_learner
                            if hasattr(model, 'forward'):
                                output = model.forward(x)
                            else:
                                output = model(x)
                        
                        # Ensure output is a tensor
                        if not isinstance(output, torch.Tensor):
                            output = torch.tensor(output, dtype=torch.float32, device=self.device)
                            
                        base_outputs.append(output)
                    except Exception as e:
                        logger.warning(f"Error getting predictions from model {i} ({self.model_names[i] if i < len(self.model_names) else 'unknown'}): {str(e)}")
                        
                # Stack outputs from base models
                if base_outputs:
                    # Stack along feature dimension
                    stacked = torch.cat(base_outputs, dim=1)
                    
                    # Forward through meta-learner
                    meta_output = self.meta_learner(stacked)
                    return meta_output
                else:
                    # Fallback to weighted average if no base outputs
                    return self._weighted_average(x)
            except Exception as e:
                logger.error(f"Error in meta-learner forward pass: {str(e)}")
                logger.error(traceback.format_exc())
                # Fallback to weighted average
                return self._weighted_average(x)
        else:
            # Use weighted average
            return self._weighted_average(x)
            
    def _weighted_average(self, x):
        """Compute weighted average of model outputs."""
        outputs = []
        valid_indices = []
        
        for i, model in enumerate(self.models):
            try:
                # Forward pass - handle meta-learner case
                is_meta_learner = (model is self.meta_learner)
                if not is_meta_learner:  # Skip meta-learner in weighted average
                    with torch.no_grad():
                        # Direct forward call without accessing meta_learner
                        if hasattr(model, 'forward'):
                            output = model.forward(x)
                        else:
                            output = model(x)
                    
                    # Ensure output is a tensor
                    if not isinstance(output, torch.Tensor):
                        output = torch.tensor(output, dtype=torch.float32, device=self.device)
                        
                    # Ensure output has right shape [batch_size, 1]
                    if output.dim() == 1:
                        output = output.unsqueeze(1)
                        
                    outputs.append(output)
                    valid_indices.append(i)
            except Exception as e:
                logger.error(f"Error in model {i} ({self.model_names[i] if i < len(self.model_names) else 'unknown'}) forward pass: {str(e)}")
                # Skip this model
                
        if not outputs:
            # If all models failed, return zeros
            return torch.zeros(x.size(0), 1, device=self.device)
            
        # Get weights for valid models
        if valid_indices:
            valid_weights = self.weights[valid_indices]
            # Renormalize
            valid_weights = valid_weights / valid_weights.sum()
        else:
            # Equal weights if all indices invalid
            valid_weights = torch.ones(len(outputs), device=self.device) / len(outputs)
            
        # Stack outputs [batch_size, n_valid_models]
        stacked = torch.cat([out for out in outputs], dim=1)
        
        # Apply weights: [batch_size, n_valid_models] * [n_valid_models]
        weighted_sum = torch.matmul(stacked, valid_weights)
        
        # Ensure output has right shape [batch_size, 1]
        if weighted_sum.dim() == 1:
            weighted_sum = weighted_sum.unsqueeze(1)
            
        return weighted_sum
        
    def eval(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            if hasattr(model, 'eval'):
                model.eval()
        return self
        
    def train(self, mode=True):
        """Set all models to training or evaluation mode."""
        self.training = mode
        for model in self.models:
            if hasattr(model, 'train'):
                if callable(model.train):
                    model.train(mode)
        return self

# Helper functions for working with the adapter

def create_adapter(
    model_type: str,
    input_dim: int,
    output_dim: int = 1,
    device: str = "cpu",
    **kwargs
) -> ModelAdapter:
    """Create a model adapter."""
    return ModelAdapter(model_type, input_dim, output_dim, device, **kwargs)

def create_adapter_ensemble(
    input_dim: int,
    output_dim: int = 1,
    ensemble_size: int = 5,
    device: str = "cpu"
) -> Tuple[List[ModelAdapter], List[str]]:
    """Create a default ensemble of model adapters."""
    # Define model types for ensemble
    model_types = ["lr", "svc", "rf", "xgb", "lgbm"]
    
    # Ensure we have enough model types
    if len(model_types) < ensemble_size:
        # Repeat model types if needed
        model_types = model_types * (ensemble_size // len(model_types) + 1)
    
    # Truncate to requested size
    model_types = model_types[:ensemble_size]
    
    # Create adapters
    adapters = []
    model_names = []
    
    for model_type in model_types:
        try:
            adapter = create_adapter(model_type, input_dim, output_dim, device)
            adapters.append(adapter)
            model_names.append(model_type)
        except Exception as e:
            logger.error(f"Error creating {model_type} adapter: {str(e)}")
            # Skip this model
    
    # If no adapters were created, create at least one simple model
    if not adapters:
        adapters.append(create_adapter("simple", input_dim, output_dim, device))
        model_names.append("simple")
        
    # Add meta-learner if we have more than one model
    if len(adapters) > 1:
        try:
            meta = create_adapter("meta_lr", len(adapters), output_dim, device)
            adapters.append(meta)
            model_names.append("meta_lr")
        except Exception as e:
            logger.error(f"Error creating meta-learner: {str(e)}")
            # Continue without meta-learner
    
    return adapters, model_names

# Fix for model_adapter.py
def create_adapter_from_config(params: Dict[str, Any], device: str = "cpu") -> ModelAdapter:
    """Create a model adapter from configuration."""
    try:
        # Extract required parameters
        model_type = params.get("model_type", params.get("estimator", "simple"))
        input_dim = params.get("input_dim", 10)
        output_dim = params.get("output_dim", 1)
        
        # Create a copy of params without duplicative keys to avoid duplicate arguments
        params_copy = params.copy()
        for key in ["model_type", "estimator", "input_dim", "output_dim"]:
            if key in params_copy:
                del params_copy[key]
                
        # Create adapter
        adapter = create_adapter(model_type, input_dim, output_dim, device, **params_copy)
        
        # Set parameters
        adapter.set_parameters(params)
            
        return adapter
    except Exception as e:
        logger.error(f"Error creating adapter from config: {str(e)}")
        logger.error(traceback.format_exc())
        # Fall back to simple model
        return create_adapter("simple", params.get("input_dim", 10), params.get("output_dim", 1), device)

def create_adapter_ensemble_from_config(
    config: List[Dict[str, Any]],
    input_dim: int,
    output_dim: int = 1,
    device: str = "cpu"
) -> Tuple[List[ModelAdapter], List[str]]:
    """Create an ensemble of model adapters from configuration."""
    adapters = []
    model_names = []
    
    # First pass: create all models except meta_lr
    for model_config in config:
        try:
            # Fix input dimension if needed
            model_config["input_dim"] = input_dim
            model_config["output_dim"] = output_dim
            
            # Get model type
            model_type = model_config.get("model_type", "simple")
            
            # Skip meta-learner for now
            if model_type == "meta_lr":
                continue
                
            # Create adapter
            adapter = create_adapter_from_config(model_config, device)
            adapters.append(adapter)
            model_names.append(model_type)
        except Exception as e:
            logger.error(f"Error creating adapter from config: {str(e)}")
            logger.error(traceback.format_exc())
            # Skip this model
    
    # If no adapters were created, create at least one simple model
    if not adapters:
        adapters.append(create_adapter("simple", input_dim, output_dim, device))
        model_names.append("simple")
    
    # Second pass: create meta-learner if it exists in config
    for model_config in config:
        model_type = model_config.get("model_type", "")
        if model_type == "meta_lr":
            try:
                # Number of base models is input_dim for meta-learner
                meta_input_dim = len(adapters)  # Number of base models
                if meta_input_dim > 0:
                    model_config["input_dim"] = meta_input_dim
                    model_config["output_dim"] = output_dim
                    
                    # Create meta-learner
                    meta = create_adapter_from_config(model_config, device)
                    adapters.append(meta)
                    model_names.append("meta_lr")
                else:
                    logger.warning("Cannot create meta-learner without base models")
            except Exception as e:
                logger.error(f"Error creating meta-learner: {str(e)}")
                logger.error(traceback.format_exc())
                # Skip meta-learner
            break  # Only create one meta-learner
        
    return adapters, model_names