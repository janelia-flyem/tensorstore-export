"""
Main entry point for Cloud Run Job deployment.

This is a batch processing job that processes DVID shard files and converts
them to Neuroglancer precomputed format using TensorStore.
"""

import sys
import asyncio
import logging
import structlog
from src.worker import main as worker_main

# Setup structured logging for Cloud Run
def setup_logging():
    """Configure structured logging for Cloud Run Jobs."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()  # JSON format for Cloud Logging
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set root logger level
    logging.basicConfig(level=logging.INFO)


async def main():
    """
    Main entry point for the Cloud Run Job.
    
    This function runs the worker and exits with appropriate status codes:
    - 0: Success (job completed normally)
    - 1: Error (job failed)
    """
    setup_logging()
    logger = structlog.get_logger()
    
    try:
        logger.info("Starting TensorStore DVID Export Worker")
        
        # Run the worker main function
        await worker_main()
        
        logger.info("Worker completed successfully")
        sys.exit(0)  # Success
        
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(0)  # Treat interruption as success
        
    except Exception as e:
        logger.error("Worker failed with exception", error=str(e), exc_info=True)
        sys.exit(1)  # Error


if __name__ == "__main__":
    asyncio.run(main())