# src/deployment/Dockerfile

# Step 1: Use Python slim image
FROM python:3.8-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the deployment requirements file into the container
# Use a lightweight requirements file for faster builds
COPY ../../requirements.txt .

# Step 4: Upgrade pip and install packages with extended timeout
# Install packages one by one to avoid timeout issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 numpy==1.24.4 && \
    pip install --no-cache-dir --timeout=1000 pandas==2.0.3 && \
    pip install --no-cache-dir --timeout=1000 scikit-learn==1.3.2 && \
    pip install --no-cache-dir --timeout=1000 pydantic==2.5.3 && \
    pip install --no-cache-dir --timeout=1000 fastapi==0.109.0 && \
    pip install --no-cache-dir --timeout=1000 uvicorn==0.27.0 && \
    pip install --no-cache-dir --timeout=1000 mlflow==2.9.2

# Step 5: Copy mlruns directory (contains trained models)
COPY mlruns/ /app/mlruns/

# Step 6: Copy only the deployment app files into the container at /app
# (source files are under src/deployment in the repo)
COPY src/deployment/*.py .

# Step 7: Expose the port the app runs on
EXPOSE 8000

# Step 8: Define the command to run the app
# We use uvicorn to serve the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]