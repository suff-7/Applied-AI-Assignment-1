import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io

import os
import requests

import os
def drive_download_gdown(file_id, destination):
    import subprocess
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        import gdown
    except ImportError:
        subprocess.check_call(['pip', 'install', 'gdown'])
        import gdown
    gdown.download(url, destination, quiet=False)

model_path = "quantity_model.pth"
gdrive_id = "1gUZbrrdW7KEY-oVfagK8GszUdcgXSq_i"

if not os.path.exists(model_path):
    drive_download_gdown(gdrive_id, model_path)


# ----- Model definition (same as before) -----
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = torch.nn.BatchNorm2d(out_channels)
        self.relu  = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = torch.nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ImprovedQuantityPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )
    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c, 1))
        return torch.nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x); x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
# ----- Model loading -----
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = ImprovedQuantityPredictor()
    # Disable weights_only to load full checkpoint safely (trust your file!)
    checkpoint = torch.load("quantity_model.pth", map_location=device, weights_only=False)
    # Extract weights if saved as state_dict or checkpoint dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- UI Layout -----
st.set_page_config(page_title="Bin Image Quantity Estimator", layout="wide")
st.markdown("""
    <h2 style='text-align: center;'>Amazon Bin Image Quantity Predictor</h2>
    <p style='text-align: center;'>Upload one or more bin images below. The model will predict object quantity and show confidence for each.</p>
    """, unsafe_allow_html=True)

st.sidebar.header("Batch Upload")
uploaded_files = st.sidebar.file_uploader("Upload bin images (.jpg, .png)", type=['jpg', 'png'], accept_multiple_files=True)

confidence_thresholds = [0.5, 1.0, 2.0]
selected_confidence_tol = st.sidebar.selectbox(
    "What range is considered 'Confident' prediction?",
    confidence_thresholds, format_func=lambda x: f"±{int(x)} objects"
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} image(s) uploaded.")
    cols = st.columns(min(5, len(uploaded_files)))
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % len(cols)]:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_qty = output.item()
            # Confidence visualization
            # For demo, we simulate confidence: If prediction is within ±selected_tol of an integer, consider "high confidence"
            close_to_int = abs(predicted_qty - round(predicted_qty)) <= selected_confidence_tol
            conf_color = "green" if close_to_int else "orange"
            st.markdown(
                f"<div style='color:{conf_color}; font-size:1.25em;'>"
                f"Predicted Quantity: <b>{predicted_qty:.2f}</b><br>"
                f"Model confidence: <b>{'High' if close_to_int else 'Low'}</b> "
                f"(within ±{int(selected_confidence_tol)} of integer)"
                "</div>",
                unsafe_allow_html=True
            )

else:
    st.warning("No image uploaded yet. Please select one or more files from the sidebar.")

st.markdown("""
    <hr>
    <div style='text-align:center; font-size:0.9em;'>
    <b>Assignment Demo</b> | Model: ResNet-style Regression | Streamlit UI | Batch prediction & confidence
    </div>
    """, unsafe_allow_html=True)



