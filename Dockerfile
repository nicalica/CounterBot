# Python image to use.
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir --timeout 120 -r requirements.txt

# Run main.py with gunicorn when the container launches
ENTRYPOINT ["gunicorn", "main:app"]

# Optional commands to pass to gunicorn
# CMD ["-b", "0.0.0.0:8080", "-t", "3600"]

