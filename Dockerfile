# Use a lightweight Python Linux image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install System Dependencies
# We absolutely need 'git' to install Chronos from GitHub
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements first (for better caching)
COPY requirements.txt .

# 3. Install Python libraries
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your app code
COPY . .

# 5. Expose the port Hugging Face expects (7860)
EXPOSE 7860

# 6. Run Streamlit on Port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]