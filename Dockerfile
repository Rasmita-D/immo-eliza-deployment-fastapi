# Starts from the python 3.10 official docker image
FROM python:3.12-slim

# Create a folder "api" at the root of the image
RUN mkdir /app

# Define /app as the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files in the current directory in /app
COPY main.py .
COPY models/ ./models/

# Run the app
# Set host to 0.0.0.0 to make it run on the container's network
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]