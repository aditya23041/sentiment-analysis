$ErrorActionPreference = "Stop"

Write-Host "Starting Our System on Port 8000..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd 'd:\Projects\sentiment anylsis'; .\.venv\Scripts\activate; python -m sentiment_analysis.cli.main serve --port 8000"

Write-Host "Starting Agentic-Semantic-Router on Port 8001..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd 'd:\Projects\Agentic-Semantic-Router'; .\.venv\Scripts\activate; uvicorn src.emotion_analysis.app.main:app --port 8001"

Write-Host "Both servers started in separate windows."
