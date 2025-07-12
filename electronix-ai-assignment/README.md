# Electronix AI Assignment

## Overview
This project fine-tunes a BERT model for sentiment analysis on the IMDb dataset and deploys it as a FastAPI service. It is containerized using Docker for CPU-only compatibility, with optional GPU support. The project was developed and tested on a Windows machine with a 4-core CPU, achieving a fine-tuning time of 32.42 minutes.

## Project Structure
```
electronix-ai-assignment/
├── fine_tune.py          # Script to fine-tune BERT model
├── api.py                # FastAPI application for sentiment prediction
├── Dockerfile            # Docker configuration for containerization
├── docker-compose.yml    # Docker Compose configuration for API deployment
├── requirements.txt      # Pinned dependencies for reproducibility
├── README.md             # Project documentation (this file)
├── model/                # Directory for fine-tuned model weights
└── data/                 # Directory for dataset (optional, empty)
```

## Setup & Run Instructions
Follow these steps to set up and run the project. All commands assume you are in the project directory (e.g., `C:\Users\vidit\Desktop\new_asss\electronix-ai-assignment`).

### 1. Prerequisites
- **Python**: Version 3.9+ (tested with Python 3.11).
- **Docker**: Docker Desktop installed (for containerized deployment, optional).
- **Git**: For cloning and pushing to GitHub.

### 2. Clone Repository
Clone the project repository from GitHub. Replace `<repository-url>` with your GitHub repository URL (e.g., `https://github.com/your-username/electronix-ai-assignment.git`):
```bash
git clone <repository-url>
cd electronix-ai-assignment
```

### 3. Create Virtual Environment
Create and activate a virtual environment to isolate dependencies:
```bash
python -m venv electronix-ai-venv
.\electronix-ai-venv\Scripts\activate
```

### 4. Install Dependencies
Install the pinned dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Dependencies Table**:
| Package         | Version | Purpose                                      |
|-----------------|---------|----------------------------------------------|
| torch           | 2.3.1   | PyTorch for model training and inference     |
| transformers    | 4.44.2  | Hugging Face Transformers for BERT model     |
| datasets        | 2.20.0  | Hugging Face Datasets for IMDb dataset       |
| fastapi         | 0.115.0 | FastAPI for async API serving                |
| uvicorn         | 0.30.6  | ASGI server for running FastAPI              |
| pydantic        | 2.8.2   | Data validation for API requests             |
| numpy           | 1.26.4  | Numerical computations (pinned for compatibility) |
| scipy           | 1.13.1  | Scientific computations (pinned for compatibility) |

### 5. Fine-Tune Model
Run the fine-tuning script to train the BERT model on a subset of the IMDb dataset (1000 training, 200 evaluation samples):
```bash
python fine_tune.py
```
- **Output**: Model weights and tokenizer are saved to the `./model` directory.
- **Fine-Tuning Time**: 32.42 minutes on a 4-core CPU (no GPU tested).

### 6. Run API Locally
Run the FastAPI server using `uvicorn`:
```bash
uvicorn api:app --host 127.0.0.1 --port 8001
```
- **Output**: API runs at `http://127.0.0.1:8001`.
- Ensure the `./model` directory contains the fine-tuned weights before running.

**Alternative (Docker)**:
Build and run the API using Docker (optional, requires Docker Desktop):
```bash
docker-compose up --build
```
- **Output**: API runs at `http://127.0.0.1:8001`.

### 7. Test API
Test the `/predict` endpoint with a `curl` command:
```bash
curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d '{"texts": ["I love this movie!", "This was terrible."]}'
```
- **Expected Response**:
  ```json
  {"predictions": ["positive", "negative"]}
  ```
- **Swagger UI**: Access at `http://127.0.0.1:8001/docs` for interactive API testing.

## Design Decisions
- **Framework**: Hugging Face Transformers was chosen for fine-tuning BERT due to its robust pre-trained models and ecosystem. PyTorch (`torch==2.3.1`) was selected for flexibility in model training.
- **API**: FastAPI was used for its asynchronous support and automatic OpenAPI documentation, enabling high-throughput inference.
- **Docker**: A slim Python 3.9 image minimizes container size, and Docker Compose simplifies deployment.
- **Async Batching**: Implemented in the `/predict` endpoint to process multiple inputs efficiently, improving throughput.
- **Dependencies**: Pinned versions (`numpy==1.26.4`, `scipy==1.13.1`, `transformers==4.44.2`, `torch==2.3.1`) resolve compatibility issues with NumPy 2.x, SciPy, and PyTorch. Removed `tf-keras` to avoid TensorFlow DLL errors.
- **Dataset**: Used a subset of the IMDb dataset (1000 training, 200 evaluation samples) to reduce fine-tuning time for demo purposes.

## Fine-Tuning Results
| Metric             | Value                     |
|--------------------|---------------------------|
| Fine-Tuning Time   | 32.42 minutes (CPU)       |
| Device             | 4-core CPU (no GPU)       |
| Training Samples   | 1000 (IMDb dataset subset)|
| Evaluation Samples | 200 (IMDb dataset subset) |

## API Documentation
- **Endpoint**: `POST /predict`
- **Request Body**:
  ```json
  {
    "texts": ["string", "string", ...]
  }
  ```
- **Response**:
  ```json
  {
    "predictions": ["positive", "negative", ...]
  }
  ```
- **Swagger UI**: Available at `http://127.0.0.1:8001/docs`.

## Troubleshooting
- **Dependency Issues**:
  - Verify dependencies in the virtual environment:
    ```bash
    pip list | findstr "torch transformers numpy scipy"
    ```
  - Reinstall if needed:
    ```bash
    pip install -r requirements.txt
    ```
- **TensorFlow DLL Error**: Resolved by using a clean virtual environment (`electronix-ai-venv`) and removing TensorFlow dependencies.
- **PyTorch Version**: Updated to `torch==2.3.1` to match available versions for Python 3.11 on Windows.
- **Model Weights**: Ensure the `./model` directory contains `pytorch_model.bin`, `config.json`, and tokenizer files before running the API.

## Notes
- **Dataset**: The model is fine-tuned on a subset of the IMDb dataset for faster training. For production, increase the dataset size in `fine_tune.py`.
- **Deployment**: Cloud deployment (e.g., Heroku, AWS) was not feasible due to time constraints. The API was tested locally at `http://127.0.0.1:8001`.
- **Environment**: Tested on Windows with Python 3.11 in a clean virtual environment (`electronix-ai-venv`).
- **Paths**: Replace `<repository-url>` with your GitHub repository URL in the setup instructions.
- **Submission**: Includes GitHub repository link, unlisted YouTube video (demonstrating fine-tuning, API, and testing), and local API testing note.