import time
import uuid
from typing import Callable
from fastapi import Request, Response
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class LoggingMiddleware:
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Log request details
        logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
        
        # Process the request and get response
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Add custom headers to response
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            # Log response details
            logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request {request_id} failed: {str(e)} in {process_time:.4f}s")
            raise