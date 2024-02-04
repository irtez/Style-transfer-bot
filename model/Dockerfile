FROM python:3.9.10

# Install torch
RUN pip3 --timeout=300 --no-cache-dir install torch torchvision

# Install other packages
COPY ./requirements.txt .
RUN pip3 --timeout=300 --no-cache-dir install -r requirements.txt

# Copy model files
COPY ./models /models

# Copy app files
COPY ./app /app
WORKDIR /app/
ENV PYTHONPATH=/app
RUN ls -lah /app/*

COPY ./start.sh /start.sh
RUN sed -i -e 's/\r$//' /start.sh
RUN chmod +x /start.sh

EXPOSE 80
CMD ["/start.sh"]