FROM python:3.10-slim

# Install Node.js (for MCP) and system tools for ADK Web
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create folder for MCP to manage
RUN mkdir -p /app/mcp_files

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Expose the default ADK Web Server port
EXPOSE 8000

# Run the ADK web server command
# --host 0.0.0.0 is critical for Docker accessibility
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8000"]
