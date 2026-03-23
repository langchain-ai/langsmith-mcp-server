FROM python:3.14-alpine

# Install build dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev

# Install uv - fast Python package installer and resolver
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Copy source code
COPY langsmith_mcp_server/ langsmith_mcp_server/

# Install the package using uv (creates .venv by default)
RUN uv sync --no-dev

# Run as non-root user for security
RUN adduser -D -u 1000 mcp && \
    chown -R mcp:mcp /app

USER mcp

# Expose port 8000 for HTTP server
EXPOSE 8000

# Run the server with uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "langsmith_mcp_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
