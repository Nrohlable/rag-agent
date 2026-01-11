
# Use official Python runtime as a parent image
FROM python:3.11-slim

# Create entrypoint
RUN mkdir /app
WORKDIR /app
RUN touch __init__.py
COPY requirements.txt /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app
EXPOSE 8000

# Run chainlit app
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8002", "--server.address=0.0.0.0"]