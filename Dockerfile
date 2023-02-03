# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt ./

# Install the required dependencies
RUN pip install -r requirements.txt

# Copy the remaining application files to the container
COPY . .

# Set the environment variable for Streamlit to bind to all IPs
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose port 8000 for FastAPI and port 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# Start the FastAPI backend and Streamlit frontend servers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["streamlit", "run", "index.py"]