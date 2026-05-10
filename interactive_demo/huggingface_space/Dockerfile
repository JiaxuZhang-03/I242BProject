FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CHECKPOINT_PATH=/app/best_model.pt
ENV DEVICE=cpu

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY static /app/static
COPY src /app/src
COPY best_model.pt /app/best_model.pt

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860", "--device", "cpu"]
