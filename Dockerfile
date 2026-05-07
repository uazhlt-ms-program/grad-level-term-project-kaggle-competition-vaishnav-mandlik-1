FROM python:3.10-slim

LABEL author="Vaishnav Mandlik"
LABEL description="LING 539 text classification pipeline"

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY classify.py ./

# Default: generate submission
CMD ["python", "classify.py", "--submit-only"]