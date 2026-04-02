import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import timm

# ---------------- DEVICE ----------------
device = "cpu"

# ---------------- MODEL ----------------
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 28)

# Load trained weights
model.load_state_dict(torch.load("efficientnet.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- GRAD-CAM ----------------
def grad_cam(model, img):
    img.requires_grad = True

    output = model(img)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    grad = img.grad[0].cpu().numpy()
    heatmap = grad.mean(axis=0)

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return cv2.resize(heatmap, (224, 224))

# ---------------- HEATMAP OVERLAY ----------------
def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    return overlay

# ---------------- MASK ----------------
def heatmap_to_mask(heatmap):
    thresh = np.percentile((heatmap * 255).astype("uint8"), 85)
    mask = ((heatmap * 255) >= thresh).astype("uint8") * 255
    return mask

# ---------------- BOUNDING BOX ----------------
def mask_to_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]

# ---------------- DRAW BOX ----------------
def draw_boxes(image, boxes):
    img = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

# ---------------- STREAMLIT UI ----------------
st.title("🌿 Plant Disease Detection System")

file = st.file_uploader("Upload Leaf Image")

if file:
    image = Image.open(file)
    img_np = np.array(image)

    st.image(img_np, caption="Original Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img_np, (224, 224)) / 255.0
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Grad-CAM
    heatmap = grad_cam(model, img_tensor)
    overlay = overlay_heatmap(img_np, heatmap)

    st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)

    # Segmentation mask
    mask = heatmap_to_mask(heatmap)
    st.image(mask, caption="Segmented Disease Area", use_column_width=True)

    # Detection boxes
    boxes = mask_to_bbox(mask)
    boxed = draw_boxes(img_np, boxes)

    st.image(boxed, caption="Detected Regions", use_column_width=True)

    # Severity
    severity = (mask > 0).sum() / mask.size * 100
    st.success(f"Severity: {severity:.2f}%")
