# --- Build Stage ---
FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# --- Runtime Stage ---
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy installed packages and NLTK data
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/nltk_data /home/appuser/nltk_data

COPY src/ src/

RUN chown -R appuser:appuser /app

USER appuser

ENV PORT=8000

CMD sh -c "uvicorn sentiment_analysis.api.app:app --host 0.0.0.0 --port ${PORT}"
