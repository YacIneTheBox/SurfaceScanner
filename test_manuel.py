import onnxruntime as ort
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from PIL import Image, ImageOps

# 1. Configurer les classes (Mets EXACTEMENT les 23 classes dans l'ordre alphabétique de tes dossiers)
CLASSES = []

NO_USE = ["food","hair","mirror","wallpaper","water"]

with open('minc-2500/categories.txt') as file:
    for line in file:
        cline = line.strip()
        if (cline not in NO_USE):
            CLASSES.append(cline)
# (Vérifie que la liste correspond bien aux dossiers de ton dataset)

# 2. Charger le modèle
ort_session = ort.InferenceSession("material_resnet_model.onnx")

def predir_image(image_path):
# A. Charger et préparer l'image (comme dans l'API)
    # 1. On charge l'image brute
    img = Image.open(image_path)

    # 2. On applique la rotation dictée par le téléphone (C'EST LA LIGNE MAGIQUE)
    img = ImageOps.exif_transpose(img)

    # 3. On la convertit en RGB
    img = img.convert('RGB')


    transform_test = transforms.Compose([
    transforms.Resize(256),       # Redimensionne le plus petit côté à 256
    transforms.CenterCrop(224),   # Coupe un carré parfait de 224x224 au centre
    ])
    img = transform_test(img)
    
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

def predir_image_tta(image_path):
    # 1. Charger l'image avec correction de rotation
    img_originale = Image.open(image_path)
    img_originale = ImageOps.exif_transpose(img_originale).convert('RGB')
    
    # 2. Remplacer CenterCrop par FiveCrop
    transform_tta = transforms.Compose([
        transforms.Resize(256),       
        transforms.FiveCrop(224), # Découpe 5 carrés !
    ])
    
    # "crops" est maintenant un tuple qui contient 5 images
    crops = transform_tta(img_originale) 
    
    # On prépare un tableau vide pour additionner les scores des 5 prédictions
    probabilites_totales = np.zeros(len(CLASSES), dtype=np.float32)
    
    # 3. Boucle sur les 5 morceaux d'image
    for crop in crops:
        # A. Préparation classique du morceau
        img_array = np.array(crop).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        # B. Inférence
        outputs = ort_session.run(None, {'input': img_array})
        
        # C. Calcul du pourcentage (Softmax)
        x = outputs[0][0]
        e_x = np.exp(x - np.max(x))
        probs = e_x / e_x.sum()
        
        # D. On ajoute ce pourcentage au score total
        probabilites_totales += probs
        
    # 4. On fait la moyenne (on divise la somme par 5)
    probabilites_moyennes = probabilites_totales / 5.0
    
    # 5. Affichage des résultats lissés
    top3_idx = np.argsort(probabilites_moyennes)[-3:][::-1]
    
    print(f"\n--- Analyse TTA de {os.path.basename(image_path)} ---")
    for idx in top3_idx:
        print(f"{CLASSES[idx].capitalize()} : {probabilites_moyennes[idx]*100:.2f}%")

# 3. Teste avec tes propres photos !

# Le chemin vers ton dossier contenant les images
dossier_tests = 'manuelTests'

# Parcourir tous les fichiers dans le dossier
for nom_fichier in os.listdir(dossier_tests):
    # Vérifier que c'est bien une image (pour éviter les erreurs avec d'autres fichiers)
    if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Créer le chemin complet : "manuelTests/nom_de_l_image.jpg"
        chemin_complet = os.path.join(dossier_tests, nom_fichier)
        
        # Lancer la prédiction
        predir_image_tta(chemin_complet)