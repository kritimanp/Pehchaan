# Start from a Python 3.9 base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project (code, model folder) into the container
COPY . .

# Expose port 7860 (Hugging Face's required port)
EXPOSE 7860

# The command to run your API on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]