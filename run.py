#!/usr/bin/env python3
import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=True,
        log_level="info"
    )
