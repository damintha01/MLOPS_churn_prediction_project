# src/deployment/Dockerfile

# Step 1: Use an official Python runtime as a parent image
FROM python:3.8-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements file from the PROJECT ROOT into the container
COPY requirements.txt .

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the deployment app files into the container
COPY src/deployment/ .

# Step 6: Expose the port the app runs on
EXPOSE 8000

# Step 7: Define the command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]