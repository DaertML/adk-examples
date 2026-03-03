FROM python:3.11-slim
# Add any libraries the agent might want to use
RUN pip install numpy pandas
WORKDIR /data
# Keep the container alive so we don't have to restart it every time
CMD ["tail", "-f", "/dev/null"]
