from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import io
import os
from torchvision import transforms
from datetime import datetime

app = FastAPI(title="Material Detector API")

SAVE_DIR = "SavedPictures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. INITIALISATION (Chargé une seule fois au démarrage)
# ==========================================
# Charger les classes dynamiquement (les 18 matériaux)
CLASSES = []
NO_USE = ["food", "hair", "mirror", "wallpaper", "water"]

try:
    with open('minc-2500/categories.txt') as file:
        for line in file:
            cline = line.strip()
            if cline not in NO_USE:
                CLASSES.append(cline)
except FileNotFoundError:
    print("ATTENTION : Le fichier categories.txt est introuvable. L'API risque de planter.")

# Charger le modèle ONNX
print("Chargement du modèle ONNX en mémoire...")
ort_session = ort.InferenceSession("material_resnet_model.onnx")

# Préparer le pipeline de transformation (Le même que pour tes tests manuels !)
transform_tta = transforms.Compose([
    transforms.Resize(256),       
    transforms.FiveCrop(224), 
])

# ==========================================
# 2. LOGIQUE DE PRÉDICTION
# ==========================================
@app.post("/predict")
async def predict_material(file: UploadFile = File(...)):
    try:
        # A. Lire l'image envoyée par la requête HTTP
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # B. Correction EXIF indispensable pour les photos de téléphone
        img = ImageOps.exif_transpose(img).convert('RGB')
        
        # C. Appliquer le Test-Time Augmentation (FiveCrop)
        crops = transform_tta(img)
        probabilites_totales = np.zeros(len(CLASSES), dtype=np.float32)
        
        # Constantes de normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        for crop in crops:
            img_array = np.array(crop).astype(np.float32) / 255.0
            img_array = (img_array - mean) / std
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)
            
            # Inférence ONNX
            outputs = ort_session.run(None, {'input': img_array})
            
            # Calcul des pourcentages (Softmax)
            x = outputs[0][0]
            e_x = np.exp(x - np.max(x))
            probs = e_x / e_x.sum()
            
            probabilites_totales += probs
            
        # D. Moyenne des 5 crops
        probabilites_moyennes = probabilites_totales / 5.0
        
        # E. Trouver le meilleur résultat
        best_idx = int(np.argmax(probabilites_moyennes))
        materiau_predit = CLASSES[best_idx] # On récupère le nom du matériau
        
        # --- NOUVEAU BLOC : SAUVEGARDE DE L'IMAGE ---
        # On génère un timestamp (ex: 20260523_143022)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # On crée le nom du fichier (ex: wood_20260523_143022.jpg)
        filename = f"{materiau_predit}_{timestamp}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        
        # On sauvegarde l'image (img est déjà chargée en mémoire au début de la fonction)
        img.save(save_path, "JPEG")
        
        # F. Renvoyer la réponse 
        return JSONResponse(content={
            "success": True,
            "materiau": CLASSES[best_idx].capitalize(),
            "certitude": float(probabilites_moyennes[best_idx]),
            # On renvoie aussi le top 3 au cas où tu voudrais afficher "Autres possibilités" dans l'UI
            "top3": [
                {"materiau": CLASSES[i].capitalize(), "score": float(probabilites_moyennes[i])} 
                for i in np.argsort(probabilites_moyennes)[-3:][::-1]
            ]
        })
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "L'API de détection de matériaux est en ligne !"}