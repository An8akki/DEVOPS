import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not found.")

if HF_REPO_ID is None:
    raise ValueError("HF_REPO_ID environment variable not found.")

api = HfApi(token=HF_TOKEN)

# Create repository if it doesn't exist
create_repo(
    repo_id=HF_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    exist_ok=True,
)

print(f"Uploading model to {HF_REPO_ID}...")

# Upload trained model
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

# Upload metrics
api.upload_file(
    path_or_fileobj="metrics.json",
    path_in_repo="metrics.json",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

# Create a simple model card
readme = """
---
license: mit
library_name: scikit-learn
---

# Boston Housing Regression

This model is automatically trained and deployed using:

- GitHub Actions
- DVC
- MLflow
- Docker
- Hugging Face Hub

## Best Model

Gradient Boosting Regressor

## Outputs

- model.pkl
- metrics.json
"""

with open("README.md", "w") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

print("Model successfully deployed to Hugging Face Hub.")
