from PIL import Image
from torchvision import transforms
import sys
from PIL import Image, ImageOps
# 1. Définir le chemin vers l'image que tu veux tester
# Tu peux changer ce chemin pour tester différentes photos
IMAGE_PATH = "manuelTests/ferrari.jpg" 

def visualiser_crop(image_path):
    try:
        # 2. Charger l'image originale
        # 1. On charge l'image brute
        img = Image.open(image_path)

        # 2. On applique la rotation dictée par le téléphone (C'EST LA LIGNE MAGIQUE)
        img = ImageOps.exif_transpose(img)

        # 3. On la convertit en RGB
        img = img.convert('RGB')
        
        # 3. Appliquer EXACTEMENT tes transformations
        transform_test = transforms.Compose([
            transforms.Resize(256),       # Redimensionne le plus petit côté à 256
            transforms.CenterCrop(224),   # Coupe un carré parfait de 224x224 au centre
        ])
        
        img_transformee = transform_test(img)
        
        # 4. Afficher les dimensions dans la console pour vérifier
        print(f"Taille originale : {img.size}")
        print(f"Taille après crop : {img_transformee.size} (Ce que l'IA va voir)")
        
        # 5. Afficher l'image à l'écran (ouvre la visionneuse par défaut de ton PC)
        img_transformee.show()
        
        # Optionnel : Sauvegarder l'image coupée dans le dossier si tu veux la garder
        # img_transformee.save("resultat_crop.jpg")
        # print("Image sauvegardée sous 'resultat_crop.jpg'")
        
    except FileNotFoundError:
        print(f"Erreur : L'image '{image_path}' est introuvable.")

# Lancer la fonction
visualiser_crop(IMAGE_PATH)