# Use the official Python image
FROM python:3.7

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Heroku dynamically assigns
EXPOSE $PORT

# Run the application with Gunicorn
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:$PORT", "app:app"]
