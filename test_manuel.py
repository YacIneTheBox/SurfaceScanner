import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. Configurer les classes (Mets EXACTEMENT les 23 classes dans l'ordre alphabétique de tes dossiers)
CLASSES = []

with open('minc-2500/categories.txt') as file:
    for line in file:
        CLASSES.append(line.strip())
# (Vérifie que la liste correspond bien aux dossiers de ton dataset)

# 2. Charger le modèle
ort_session = ort.InferenceSession("material_model.onnx")

def predir_image(image_path):
# A. Charger et préparer l'image (comme dans l'API)
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # ADD dtype=np.float32 HERE:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_array = (img_array - mean) / std
    
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    # B. Faire la prédiction
    outputs = ort_session.run(None, {'input': img_array})
    
    # C. Calculer les pourcentages (Softmax)
    x = outputs[0][0]
    e_x = np.exp(x - np.max(x))
    probabilities = e_x / e_x.sum()
    
    # D. Récupérer le top 3 des prédictions
    top3_idx = np.argsort(probabilities)[-3:][::-1]
    
    print(f"\n--- Analyse de {image_path} ---")
    for idx in top3_idx:
        print(f"{CLASSES[idx].capitalize()} : {probabilities[idx]*100:.2f}%")

# 3. Teste avec tes propres photos !

with open("manuelTests") as img:
    predir_image(img)