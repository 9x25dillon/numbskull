FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "chaos_llm.api:app", "--host", "0.0.0.0", "--port", "8000"]
