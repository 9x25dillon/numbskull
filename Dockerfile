FROM python:3.11-slim
cursor/bc-f408c7bd-bc2a-48a4-bc8d-0989f628ad52-ef2e

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.chaos_llm.api:app", "--host", "0.0.0.0", "--port", "8000"]

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "chaos_llm.api:app", "--host", "0.0.0.0", "--port", "8000"]
main
