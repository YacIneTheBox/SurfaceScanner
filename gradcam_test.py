import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

# Importation de la librairie Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 1. CONFIGURATION ET CHARGEMENT DU MODÈLE
# ==========================================
NUM_CLASSES = 18
IMAGE_PATH = "SavedPictures/skin_20260524_103231.jpg" # Remplace par une de tes photos !

# Charger les classes pour l'affichage
CLASSES = []
NO_USE = ["food", "hair", "mirror", "wallpaper", "water"]
try:
    with open('minc-2500/categories.txt') as file:
        for line in file:
            cline = line.strip()
            if cline not in NO_USE:
                CLASSES.append(cline)
except FileNotFoundError:
    print("Fichier categories.txt introuvable.")

print("Chargement du modèle PyTorch...")
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("material_resnet_model.pth", weights_only=True))
model.eval() # Mode évaluation très important !

# ==========================================
# 2. PRÉPARATION DE L'IMAGE
# ==========================================
# On charge et redresse l'image (Correction EXIF)
img_pil = Image.open(IMAGE_PATH)
img_pil = ImageOps.exif_transpose(img_pil).convert('RGB')

# On applique le crop géométrique (pour que la heatmap corresponde EXACTEMENT au tenseur)
crop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
img_cropped = crop_transform(img_pil)

# L'image pour l'affichage (valeurs entre 0 et 1)
rgb_img = np.float32(img_cropped) / 255.0

# L'image pour le modèle (Normalisée mathématiquement)
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = tensor_transform(img_cropped).unsqueeze(0) # Ajout de la dimension "Batch"

# ==========================================
# 3. GÉNÉRATION DE LA CARTE GRAD-CAM
# ==========================================
# On cible la toute dernière couche de convolution de ResNet50
target_layers = [model.layer4[-1]]

# Initialisation de Grad-CAM
with GradCAM(model=model, target_layers=target_layers) as cam:
    
    # Génération de la carte (targets=None demande d'expliquer la classe la plus probable)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    # On récupère le résultat pour la première image du batch
    grayscale_cam = grayscale_cam[0, :]
    
    # On superpose la heatmap rouge/bleue sur l'image d'origine
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# ==========================================
# 4. AFFICHAGE DES RÉSULTATS
# ==========================================
# Faire une prédiction classique pour savoir ce que l'IA a vu
with torch.no_grad():
    outputs = model(input_tensor)
    pred_idx = outputs.argmax(dim=1).item()
    pred_class = CLASSES[pred_idx]

print(f"\nL'IA pense que cette image est : {pred_class.upper()}")
print("Affichage de la zone d'attention (Ferme la fenêtre pour quitter)...")

# Conversion en format BGR pour l'affichage avec OpenCV
visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

cv2.imshow('Grad-CAM : Ce que regarde l\'IA', visualization_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()