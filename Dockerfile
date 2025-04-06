# syntax=docker/dockerfile:1

# Use a slim Python base image
FROM python:3.11-slim-bookworm AS base

# Set working directory
WORKDIR /app

# Install dependencies in a builder stage
FROM base AS builder

# Copy requirements file
COPY --link requirements.txt ./

# Install dependencies into a virtual environment
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /app/.venv

RUN apt-get update && apt-get install -y git

RUN /app/.venv/bin/pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Final stage
FROM base AS final

# Copy application code
COPY --link . .

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"

# Expose application port (if applicable)
EXPOSE 8000

# Set default command
CMD ["python", "-m examples.server_side"]