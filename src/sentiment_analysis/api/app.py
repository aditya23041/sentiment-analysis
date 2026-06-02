"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sentiment_analysis.api.routes import router

load_dotenv()

logger = logging.getLogger(__name__)

# Paths
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: pre-load models on startup."""
    logger.info("Starting Sentiment Analysis API...")

    # Pre-load VADER (always available, no API key needed)
    from sentiment_analysis.models import get_model

    get_model("vader")
    logger.info("Default model (VADER) pre-loaded")

    yield

    logger.info("Shutting down Sentiment Analysis API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Aditya's Sentiment Analyzer API",
        description=(
            "Multi-model sentiment analysis platform with "
            "TextBlob, VADER, and Transformer backends."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router, prefix="/api")

    # Static files
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Dashboard template route
    if _TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

        @app.get("/", include_in_schema=False, response_class=HTMLResponse)
        async def dashboard(request: Request) -> HTMLResponse:
            return templates.TemplateResponse(request=request, name="index.html")

    return app


# Module-level app instance for `uvicorn sentiment_analysis.api.app:app`
app = create_app()
