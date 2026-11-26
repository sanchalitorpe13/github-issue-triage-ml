"""
FastAPI Application Factory & Serving Layer.
Configures CORS, lifespan initialization, standardized exception handlers,
and registers modern API routes while maintaining backward compatibility with legacy endpoints.
"""
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from app.dependencies import get_triage_engine
from app.routes.health import router as health_router
from app.routes.triage import router as triage_router
from app.routes.model_info import router as model_info_router
from src.triage_ml.utils.logger import get_logger

logger = get_logger("app_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring model and triage engine are loaded on startup."""
    logger.info("Initializing API application and pre-loading ML triage engine...")
    try:
        engine = get_triage_engine()
        if engine.predictor.is_ready():
            logger.info("ML Triage Engine successfully pre-loaded.")
        else:
            logger.warning("ML Triage Engine loaded without active model artifacts.")
    except Exception as e:
        logger.error(f"Failed to initialize triage engine during startup: {e}")

    yield

    logger.info("Shutting down API application...")


def create_app() -> FastAPI:
    """
    Application factory building the configured FastAPI service.

    Returns:
        FastAPI: Production-ready application instance.
    """
    application = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 1. CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Standardized Exception Handlers
    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Uniform handler for HTTPExceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "path": request.url.path
            }
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Uniform handler for Pydantic input validation failures."""
        formatted_errors = []
        for err in exc.errors():
            loc = " -> ".join([str(x) for x in err.get("loc", [])])
            msg = err.get("msg", "Validation error")
            formatted_errors.append(f"{loc}: {msg}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "The submitted request body failed input validation.",
                "details": formatted_errors,
                "status_code": 422,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "path": request.url.path
            }
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Uniform handler for unexpected internal exceptions without leaking stack traces."""
        logger.error(f"Unhandled exception at {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected internal server error occurred while processing the request.",
                "status_code": 500,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "path": request.url.path
            }
        )

    # 3. Register Modern API Routers
    application.include_router(health_router)
    application.include_router(triage_router)
    application.include_router(model_info_router)

    # 4. Legacy Backward-Compatible Endpoint
    @application.post(
        "/predict",
        tags=["Legacy Compatibility"],
        summary="Legacy Predict Endpoint",
        description="Preserved for backward compatibility with Phase 1 REST callers."
    )
    def legacy_predict(text: str):
        """Preserves legacy /predict route returning {"predicted_labels": [category]}."""
        engine = get_triage_engine()
        res = engine.analyze_issue(text=text)
        return {
            "predicted_labels": [res.predicted_category]
        }

    return application


# Global application instance for uvicorn
app = create_app()
