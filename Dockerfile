# Paleidžia Streamlit demonstraciją su jau ištreniruotais modeliais (models/ katalogas).
#   docker build -t agresijos-filtras .
#   docker run -p 8501:8501 agresijos-filtras
#   -> http://localhost:8501
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# CPU torch versija – daug mažesnis atvaizdas nei su CUDA
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
