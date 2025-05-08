# # import torch
# # import torch.nn as nn

# # class SimpleModel(nn.Module):
# #     """
# #     Simple neural network model with configurable hidden layers
# #     """
# #     def __init__(self, input_dim, hidden_dims):
# #         super(SimpleModel, self).__init__()
        
# #         layers = []
# #         prev_dim = input_dim
        
# #         for dim in hidden_dims:
# #             layers.append(nn.Linear(prev_dim, dim))
# #             layers.append(nn.ReLU())
# #             prev_dim = dim
        
# #         layers.append(nn.Linear(prev_dim, 1))
        
# #         self.network = nn.Sequential(*layers)
    
# #     def forward(self, x):
# #         return self.network(x)

# # class FraudDetectionModel(nn.Module):
# #     """
# #     A more sophisticated model for fraud detection
# #     """
# #     def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
# #         super(FraudDetectionModel, self).__init__()
        
# #         layers = []
# #         prev_dim = input_dim
        
# #         # Create hidden layers with batch normalization and dropout
# #         for i, dim in enumerate(hidden_dims):
# #             layers.append(nn.Linear(prev_dim, dim))
# #             layers.append(nn.BatchNorm1d(dim))
# #             layers.append(nn.ReLU())
# #             layers.append(nn.Dropout(dropout_rate))
# #             prev_dim = dim
        
# #         # Output layer (binary classification)
# #         layers.append(nn.Linear(prev_dim, 1))
# #         layers.append(nn.Sigmoid())  # For binary classification
        
# #         self.network = nn.Sequential(*layers)
    
# #     def forward(self, x):
# #         return self.network(x)







# #### This is a simplified version of the model for demonstration purposes.
# # The actual model might be more complex depending on the use case.
# # In models.py
# import torch
# import torch.nn as nn

# class TinyModel(nn.Module):
#     def __init__(self, input_dim=10, output_dim=1):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         return self.linear(x)