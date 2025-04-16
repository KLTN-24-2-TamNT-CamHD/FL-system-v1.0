"""
Flower server implementation for federated learning
Current Date (UTC): 2025-04-01 13:50:21
Admin: dinhcam89
"""

from .server import start_server, FraudDetectionStrategy

__all__ = ['start_server', 'FraudDetectionStrategy']

# Server configuration
DEFAULT_PORT = 8080
MIN_CLIENTS = 2
NUM_ROUNDS = 3

# Metadata
SERVER_VERSION = '1.0.0'
LAST_UPDATED = '2025-04-01 13:50:21'
ADMIN = 'dinhcam89'