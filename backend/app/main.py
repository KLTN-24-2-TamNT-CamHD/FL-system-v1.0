from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Only import the routes we need
from app.api.routes import institutions, training
from app.config import settings

app = FastAPI(
    title="Fraud Detection FL System API",
    description="API for Federated Learning Fraud Detection System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include only essential routers
app.include_router(institutions.router, prefix="/api/institutions", tags=["Institution Management"])
app.include_router(training.router, prefix="/api/training", tags=["Training Management"])

@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "success",
        "message": "Service is running",
        "version": "1.0.0",
        "admin": "dinhcam89",  # Added admin info
        "timestamp": "2025-03-30 16:35:30"  # Current timestamp
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)