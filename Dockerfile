FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY src/ ./src/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY pytest.ini .

ENV PYTHONPATH="/app/src"

CMD ["pytest", "-v"]
