FROM python:3.9.10

# Install dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY ./app /app

# Start
CMD ["python", "main.py"]