# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./
# Copy server requirements if it exists, otherwise create empty file
RUN if [ -f requirements_server.txt ]; then cp requirements_server.txt .; else echo "# No server requirements" > requirements_server.txt; fi

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements_server.txt

# Copy application code
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY start_server.py ./
COPY hvac_config.json ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port (will be overridden by config or command line)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the server
CMD ["python", "start_server.py", "--host", "0.0.0.0"] 