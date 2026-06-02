# Sentiment Analysis Platform

A production-grade, multi-model sentiment analysis platform with a REST API, web dashboard, and CLI.

## Features

- **Multi-Model Architecture** — Switch between VADER, TextBlob, and HuggingFace Transformers
- **5-Class Sentiment** — VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE
- **REST API** — FastAPI with OpenAPI docs, batch processing, CSV upload, model comparison
- **Web Dashboard** — Dark-themed glassmorphism UI with animated gauge, charts, drag-drop CSV
- **Rich CLI** — Analyze, batch, compare, and serve commands with beautiful terminal output
- **Text Preprocessing** — URL/HTML stripping, contraction expansion, repeated char normalization
- **CSV/JSON I/O** — Analyze files and export results
- **Pydantic Schemas** — Type-safe request/response models with validation
- **Tested** — Unit tests, integration tests, API tests with pytest
- **CI/CD** — GitHub Actions, Docker support

## Quick Start

### Install

```bash
# Clone & enter the project
cd "sentiment anylsis"

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install the package (editable mode for development)
pip install -e ".[dev]"
```

### Launch the Dashboard

```bash
# Start the web server
sentiment serve

# Or directly:
uvicorn sentiment_analysis.api.app:app --reload
```

Open **http://localhost:8000** for the dashboard, **http://localhost:8000/docs** for API docs.

### CLI Usage

```bash
# Analyze a single text
sentiment analyze "I love this product!"

# Compare all models
sentiment compare "This movie was terrible"

# Batch analyze a CSV file
sentiment batch data.csv --column reviews --output results.csv

# List available models
sentiment models
```

### Use as a Library

```python
from sentiment_analysis.core.analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(model="vader")

# Single text
result = analyzer.analyze("This is amazing!")
print(result.sentiment)      # SentimentLabel.VERY_POSITIVE
print(result.polarity)       # 0.6239
print(result.confidence)     # 0.7487

# Batch
results = analyzer.analyze_batch(["Great!", "Terrible.", "Okay"])

# Compare all models
comparison = analyzer.compare_models("I love this!")
print(comparison.consensus)  # SentimentLabel.POSITIVE

# CSV file
df = analyzer.analyze_csv("reviews.csv", text_column="comment")
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & available models |
| `GET` | `/api/models` | List available models |
| `POST` | `/api/analyze` | Analyze single text |
| `POST` | `/api/analyze/batch` | Analyze multiple texts |
| `POST` | `/api/analyze/compare` | Compare all models on one text |
| `POST` | `/api/analyze/csv` | Upload & analyze CSV file |

## Project Structure

```
sentiment anylsis/
├── src/sentiment_analysis/
│   ├── models/             # NLP backends (VADER, TextBlob, Transformer)
│   ├── core/               # Analyzer, schemas, preprocessing
│   ├── api/                # FastAPI REST API
│   ├── web/                # Dashboard (templates, static)
│   └── cli/                # Click CLI
├── tests/                  # pytest test suite
├── pyproject.toml          # Modern Python packaging
├── Dockerfile              # Container deployment
└── .github/workflows/      # CI/CD
```

## Available Models

| Model | Best For | Speed | Accuracy |
|-------|----------|-------|----------|
| **VADER** | Social media, short text, emojis | ⚡ Fast | ⭐⭐⭐ |
| **TextBlob** | General purpose, simple text | ⚡ Fast | ⭐⭐ |
| **Transformer** | Maximum accuracy (requires `torch`) | 🐢 Slow | ⭐⭐⭐⭐⭐ |

Install transformer support: `pip install -e ".[transformers]"`

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/sentiment_analysis --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Docker

```bash
docker build -t sentiment-analysis .
docker run -p 8000:8000 sentiment-analysis
```

## License

Open source — feel free to modify and extend!
