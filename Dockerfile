FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000
# serve.py imports features.py from the same dir, so run from src/
WORKDIR /app/src
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
