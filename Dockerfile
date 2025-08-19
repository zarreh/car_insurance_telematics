# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Configure Poetry to not create virtual environment (install globally in container)
RUN poetry config virtualenvs.create false

# Copy Poetry configuration files
COPY pyproject.toml ./
COPY poetry.lock ./

# Install dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Copy project source code
COPY car_insurance_telematics/ ./car_insurance_telematics/
COPY Makefile ./

# Create necessary directories
RUN mkdir -p data/processed data/raw logs model_registry

# Copy data and model files if they exist
# Note: These will be mounted as volumes at runtime, so we just create the structure
RUN echo "Data and model directories created for volume mounting"

# Set Python path
ENV PYTHONPATH=/app

# Expose port (if you plan to add API endpoints later)
EXPOSE 8000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
case "$1" in\n\
  train)\n\
    shift\n\
    python -m car_insurance_telematics.modeling.train_models "$@"\n\
    ;;\n\
  infer)\n\
    shift\n\
    python -m car_insurance_telematics.modeling.run_inference "$@"\n\
    ;;\n\
  infer-sample)\n\
    python -m car_insurance_telematics.modeling.run_inference --use-sample-data\n\
    ;;\n\
  infer-batch)\n\
    python -m car_insurance_telematics.modeling.run_inference --input-file ./data/processed/processed_trips_1200_drivers.csv\n\
    ;;\n\
  lint)\n\
    autoflake car_insurance_telematics --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py\n\
    black car_insurance_telematics --line-length 120 -q\n\
    isort car_insurance_telematics\n\
    ;;\n\
  bash)\n\
    /bin/bash\n\
    ;;\n\
  *)\n\
    echo "Usage: $0 {train|infer|infer-sample|infer-batch|lint|bash}"\n\
    echo ""\n\
    echo "Commands:"\n\
    echo "  train         - Train the ML models"\n\
    echo "  infer         - Run inference with custom parameters"\n\
    echo "  infer-sample  - Run inference with sample data"\n\
    echo "  infer-batch   - Run batch inference on processed dataset"\n\
    echo "  lint          - Lint and format the code"\n\
    echo "  bash          - Open bash shell"\n\
    exit 1\n\
    ;;\n\
esac' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
