# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy local code to the container
COPY . /app

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port for any API/service if needed
EXPOSE 8000

# Define environment variable for production
ENV ENV=production

# Run the main application
CMD ["python", "main.py"]
