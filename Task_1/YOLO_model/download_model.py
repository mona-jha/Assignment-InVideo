from huggingface_hub import hf_hub_download
import shutil

# Download the TorchScript model from Hugging Face
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
print("Model downloaded to:", model_path)

# Copy the model file to your project root as "model.pt".
shutil.copy(model_path, "../model.pt")
print("Model copied to ./model.pt")


# run the the below code to download the model
"""
source venv/bin/activate
pip install -r requirements.txt
python download_model.py
deactivate

"""