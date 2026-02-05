# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8080

# Create a shell script to run BOTH Backend and Frontend
RUN echo '#!/bin/bash \n\
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 & \n\
streamlit run src/frontend/app.py --server.port 8080 --server.address 0.0.0.0 \
' > ./start.sh && chmod +x ./start.sh

# Start the app
CMD ["./start.sh"]