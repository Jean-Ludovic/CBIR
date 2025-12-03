# features.py
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Exemple avec ResNet18
_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_model.fc = torch.nn.Identity()
_model.eval()

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = _model(x).squeeze(0).numpy()
    # Optionnel : normalisation L2
    feat = feat / np.linalg.norm(feat)
    return feat
