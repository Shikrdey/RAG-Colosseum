# For HuggingFace Spaces (Docker template). Spaces expects the app on port 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    MODEL_CACHE=/home/user/app/.model_cache \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py knowledge_base.json ./

# Download the embedding model while building, so the first request is not slow.
RUN python -c "from app import get_index; get_index()"

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
