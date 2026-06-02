"""API endpoint tests using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentiment_analysis.api.app import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "available_models" in data
        assert "vader" in data["available_models"]


class TestAnalyzeEndpoint:
    def test_analyze_single_text(self, client: TestClient):
        res = client.post("/api/analyze", json={"text": "I love this!", "model": "vader"})
        assert res.status_code == 200
        data = res.json()
        assert data["polarity"] > 0
        assert data["model_used"] == "vader"

    def test_analyze_with_textblob(self, client: TestClient):
        res = client.post("/api/analyze", json={"text": "This is terrible", "model": "textblob"})
        assert res.status_code == 200
        data = res.json()
        assert data["polarity"] < 0

    def test_analyze_invalid_model(self, client: TestClient):
        res = client.post("/api/analyze", json={"text": "Hello", "model": "nonexistent"})
        assert res.status_code == 400

    def test_analyze_empty_text(self, client: TestClient):
        res = client.post("/api/analyze", json={"text": "", "model": "vader"})
        assert res.status_code == 422  # Pydantic validation


class TestBatchEndpoint:
    def test_batch_analysis(self, client: TestClient):
        res = client.post("/api/analyze/batch", json={
            "texts": ["I love this!", "This is terrible", "Okay"],
            "model": "vader",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_batch_empty_list(self, client: TestClient):
        res = client.post("/api/analyze/batch", json={"texts": [], "model": "vader"})
        assert res.status_code == 422  # Pydantic min_length validation


class TestCompareEndpoint:
    def test_compare_models(self, client: TestClient):
        res = client.post("/api/analyze/compare", json={"text": "I love this product!"})
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert len(data["results"]) >= 2

    def test_compare_empty_text(self, client: TestClient):
        res = client.post("/api/analyze/compare", json={"text": ""})
        assert res.status_code == 422


class TestModelsEndpoint:
    def test_list_models(self, client: TestClient):
        res = client.get("/api/models")
        assert res.status_code == 200
        data = res.json()
        assert "vader" in data["models"]
        assert "textblob" in data["models"]


class TestCSVEndpoint:
    def test_csv_upload(self, client: TestClient, tmp_path):
        csv_content = (
            'text,category\n'
            '"I love this!",positive\n'
            '"Terrible product",negative\n'
        )
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with open(csv_file, "rb") as f:
            res = client.post(
                "/api/analyze/csv?text_column=text&model=vader",
                files={"file": ("test.csv", f, "text/csv")},
            )
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 2

    def test_csv_non_csv_file(self, client: TestClient, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a csv")

        with open(txt_file, "rb") as f:
            res = client.post(
                "/api/analyze/csv",
                files={"file": ("test.txt", f, "text/plain")},
            )
        assert res.status_code == 400
