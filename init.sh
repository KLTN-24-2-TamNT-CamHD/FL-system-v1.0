#!/bin/bash

# Create the main backend directory
mkdir -p backend/app/api/routes backend/app/models backend/app/schemas backend/app/services backend/app/utils backend/tests

# Create __init__.py files
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/api/routes/__init__.py
touch backend/app/models/__init__.py
touch backend/app/schemas/__init__.py
touch backend/app/services/__init__.py
touch backend/app/utils/__init__.py

# Create other files
touch backend/app/main.py
touch backend/app/config.py
touch backend/app/database.py
touch backend/app/dependencies.py
touch backend/app/middleware.py

# Create route files
touch backend/app/api/routes/auth.py
touch backend/app/api/routes/institutions.py
touch backend/app/api/routes/training.py
touch backend/app/api/routes/models.py
touch backend/app/api/routes/fraud.py
touch backend/app/api/routes/blockchain.py
touch backend/app/api/routes/storage.py
touch backend/app/api/routes/analytics.py

# Create model files
touch backend/app/models/user.py
touch backend/app/models/institution.py
touch backend/app/models/training.py
touch backend/app/models/fraud.py

# Create schema files
touch backend/app/schemas/auth.py
touch backend/app/schemas/institution.py
touch backend/app/schemas/fraud.py

# Create service files
touch backend/app/services/auth_service.py
touch backend/app/services/fl_service.py
touch backend/app/services/blockchain_service.py
touch backend/app/services/ipfs_service.py

# Create utility files
touch backend/app/utils/security.py
touch backend/app/utils/logging.py

# Create other files
touch backend/requirements.txt
touch backend/Dockerfile
touch backend/.env

echo "Folder structure initialized."
