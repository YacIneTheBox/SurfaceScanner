import os
import shutil
import random

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
DOSSIER_KAGGLE = "donnees_kaggle"
DOSSIER_PERSO = "mes_photos"
DOSSIER_DATASET = "dataset"

# Combien d'images Kaggle on garde par classe (Max 25% des 2000 de MINC)
LIMITE_SYNTHETIQUE = 500 

# Le dictionnaire pour traduire les lettres de tes fichiers Kaggle en vrais noms de dossiers
MAPPING_CLASSES = {
    "m": "metal",
    "w": "wood",
    "p": "plastic",
    "g": "glass"
}

def ajouter_images_synthetiques():
    print("--- 1. Traitement des images synthétiques Kaggle ---")
    
    # On prépare un dictionnaire pour trier les fichiers trouvés
    fichiers_par_lettre = {"m": [], "w": [], "p": [], "g": []}
    
    # 1. Lire tous les fichiers du dossier Kaggle
    if not os.path.exists(DOSSIER_KAGGLE):
        print(f"Erreur : Le dossier {DOSSIER_KAGGLE} n'existe pas.")
        return
        
    for nom_fichier in os.listdir(DOSSIER_KAGGLE):
        if not nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Extraire la lettre magique (ex: "image_0012_m.jpg" -> on veut le "m")
        nom_sans_ext = os.path.splitext(nom_fichier)[0] # "image_0012_m"
        lettre = nom_sans_ext.split('_')[-1].lower()    # "m"
        
        if lettre in fichiers_par_lettre:
            fichiers_par_lettre[lettre].append(nom_fichier)
            
    # 2. Copier un échantillon au hasard pour chaque classe
    for lettre, fichiers in fichiers_par_lettre.items():
        nom_classe = MAPPING_CLASSES[lettre]
        
        # Mélanger et sélectionner les 500 premières
        random.shuffle(fichiers)
        fichiers_a_garder = fichiers[:LIMITE_SYNTHETIQUE]
        
        dossier_destination = os.path.join(DOSSIER_DATASET, "train", nom_classe)
        os.makedirs(dossier_destination, exist_ok=True)
        
        compteur = 0
        for fichier in fichiers_a_garder:
            chemin_source = os.path.join(DOSSIER_KAGGLE, fichier)
            chemin_dest = os.path.join(dossier_destination, fichier)
            
            if not os.path.exists(chemin_dest):
                shutil.copy(chemin_source, chemin_dest)
                compteur += 1
                
        print(f"[{nom_classe}] {compteur} images synthétiques ajoutées à l'entraînement.")

def ajouter_photos_personnelles():
    print("\n--- 2. Traitement de tes photos personnelles ---")
    
    if not os.path.exists(DOSSIER_PERSO):
        print(f"Erreur : Le dossier {DOSSIER_PERSO} n'existe pas.")
        return

    # Parcourir les sous-dossiers (metal, wood, etc.)
    for nom_classe in os.listdir(DOSSIER_PERSO):
        chemin_classe = os.path.join(DOSSIER_PERSO, nom_classe)
        
        if not os.path.isdir(chemin_classe):
            continue
            
        fichiers = [f for f in os.listdir(chemin_classe) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not fichiers:
            continue
            
        # Mélanger tes photos pour avoir une distribution aléatoire
        random.shuffle(fichiers)
        
        # Calculer la répartition (10% pour la validation)
        nb_val = max(1, int(len(fichiers) * 0.1)) # Au moins 1 photo en validation
        fichiers_val = fichiers[:nb_val]
        fichiers_train = fichiers[nb_val:]
        
        # Créer les dossiers de destination s'ils n'existent pas
        dest_train = os.path.join(DOSSIER_DATASET, "train", nom_classe)
        dest_val = os.path.join(DOSSIER_DATASET, "val", nom_classe)
        os.makedirs(dest_train, exist_ok=True)
        os.makedirs(dest_val, exist_ok=True)
        
        # Fonction utilitaire pour copier une liste
        def copier_liste(liste_fichiers, destination):
            c = 0
            for fichier in liste_fichiers:
                src = os.path.join(chemin_classe, fichier)
                dst = os.path.join(destination, fichier)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                    c += 1
            return c
            
        ajouts_train = copier_liste(fichiers_train, dest_train)
        ajouts_val = copier_liste(fichiers_val, dest_val)
        
        print(f"[{nom_classe}] {ajouts_train} photos envoyées en TRAIN | {ajouts_val} photos envoyées en VAL.")

# ==========================================
# EXECUTION
# ==========================================
random.seed(42) # Fixe l'aléatoire pour que le script soit prévisible si tu le relances
ajouter_images_synthetiques()
ajouter_photos_personnelles()
print("\nOpération terminée ! Ton dataset est prêt pour un nouvel entraînement.")