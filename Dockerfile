# ================================================
# Dockerfile — E-Commerce ML API
# ================================================

# Step 1: Start from official Python image (slim = smaller size)
FROM python:3.11-slim

# Step 2: Set working directory inside the container
#         All files will live here inside Docker
WORKDIR /app

# Step 3: Copy requirements first
#         Docker caches this layer — so if requirements don't change,
#         it won't re-install everything on every rebuild (saves time)
COPY requirements.txt .

# Step 4: Install all Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your entire project into the container
COPY . .

# Step 6: Tell Docker which port the app runs on
EXPOSE 8000

# Step 7: Command to start the API when container runs
#         0.0.0.0 means "accept connections from outside the container"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]