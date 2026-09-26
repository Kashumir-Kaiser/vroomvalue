FROM python:3.12-slim
ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir . \
    && useradd -r -u 10001 appuser \
    && mkdir -p /tmp/matplotlib \
    && chown -R appuser:appuser /app /tmp/matplotlib
USER appuser
EXPOSE 8000
CMD ["uvicorn", "apps.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]