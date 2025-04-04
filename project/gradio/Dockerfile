FROM ultralytics/ultralytics:latest

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Gradio code
COPY . .

# Run the Gradio interface
CMD ["python", "gradio_app.py"]
