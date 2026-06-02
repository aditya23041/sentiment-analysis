"""API route definitions for sentiment analysis endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from sentiment_analysis import __version__
from sentiment_analysis.core.analyzer import SentimentAnalyzer
from sentiment_analysis.core.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    SentimentResult,
)
from sentiment_analysis.models import list_models

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Sentiment Analysis"])

# Cache analyzer instances per model
_analyzers: dict[str, SentimentAnalyzer] = {}


def _get_analyzer(model: str) -> SentimentAnalyzer:
    """Get or create a cached analyzer for the given model."""
    if model not in _analyzers:
        try:
            _analyzers[model] = SentimentAnalyzer(model=model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
    return _analyzers[model]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and list available models."""
    return HealthResponse(
        status="ok",
        version=__version__,
        available_models=list_models(),
    )


@router.get("/models")
async def get_models() -> dict[str, list[str]]:
    """List available sentiment analysis models."""
    return {"models": list_models()}


@router.post("/analyze", response_model=SentimentResult)
async def analyze_text(request: AnalysisRequest) -> SentimentResult:
    """Analyze sentiment of a single text.

    Choose a model backend: 'textblob', 'vader', or 'transformer' (if installed).
    """
    analyzer = _get_analyzer(request.model)
    try:
        return analyzer.analyze(request.text)
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}") from None


@router.post("/analyze/batch", response_model=AnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest) -> AnalysisResponse:
    """Analyze sentiment for multiple texts in one request."""
    analyzer = _get_analyzer(request.model)
    try:
        results = analyzer.analyze_batch(request.texts)
        return AnalysisResponse(
            results=results,
            total=len(results),
            model_used=request.model,
        )
    except Exception as e:
        logger.exception("Batch analysis failed")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {e!s}") from None


@router.post("/analyze/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest) -> CompareResponse:
    """Compare results from all available models on the same text."""
    analyzer = _get_analyzer("vader")  # Use any analyzer for comparison
    try:
        return analyzer.compare_models(request.text)
    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e!s}") from None


@router.post("/analyze/csv")
async def analyze_csv(
    file: UploadFile = File(...),
    text_column: str = Query(default="text", description="Column name containing text"),
    model: str = Query(default="vader", description="Model to use"),
) -> dict:
    """Upload a CSV file and analyze sentiment for a text column.

    Returns the analysis results as a list of records.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv file")

    analyzer = _get_analyzer(model)

    try:
        content = await file.read()
        df = analyzer.analyze_csv_content(content, text_column=text_column)
        records = df.to_dict(orient="records")
        return {
            "filename": file.filename,
            "total": len(records),
            "model_used": model,
            "results": records,
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.exception("CSV analysis failed")
        raise HTTPException(status_code=500, detail=f"CSV analysis failed: {e!s}") from None
