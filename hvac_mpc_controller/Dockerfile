ARG BUILD_FROM
FROM $BUILD_FROM

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    curl \
    python3-dev \
    musl-dev

# Set working directory
WORKDIR /app

# Copy requirements from parent directory
COPY ../requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from parent directory
COPY ../src/ ./src/
COPY ../static/ ./static/
COPY ../templates/ ./templates/
COPY ../start_server.py ./
COPY generate_config.py ./
COPY run.sh ./

# Create non-root user
RUN adduser -D -s /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the server
CMD ["/app/run.sh"] 