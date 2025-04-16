"""
Federated Learning Application
Current Date (UTC): 2025-04-01 13:50:21
Admin: dinhcam89
"""

import os

# Set up package-level constants
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_ROOT, 'data')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Package version
__version__ = '1.0.0'
__author__ = 'dinhcam89'
__updated__ = '2025-04-01 13:50:21'