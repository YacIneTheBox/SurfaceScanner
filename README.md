# Material Scanner API 🔍🧱

Bienvenue sur le dépôt du backend **Material Scanner**. 

Ce projet est un outil d'intelligence artificielle conçu pour identifier rapidement la nature d'un matériau ou d'une texture de surface à partir d'une simple photographie (téléphone mobile ou autre). Il est pensé comme un outil d'aide à la décision pour l'ingénierie des surfaces et des matériaux.

*(Note : Ce dépôt contient uniquement l'infrastructure serveur Python. L'application mobile cliente n'est pas incluse).*

---

## 🚀 À propos du projet

Le système utilise un réseau de neurones **ResNet**, capable d'analyser les micro-textures, la réflectivité et les motifs géométriques de la surface pour corréler ces signatures visuelles à une base de données de matériaux de référence.

### ✨ Fonctionnalités clés
* **Haute Performance :** Le modèle a été converti au format universel **ONNX**, permettant une analyse et une réponse en quelques fractions de seconde.
* **Précision renforcée (TTA) :** Le serveur utilise le *Test-Time Augmentation* (FiveCrop). Au lieu d'analyser l'image une seule fois, l'IA l'analyse sous 5 angles/découpages différents et fait la moyenne pour garantir un résultat robuste.
* **Correction EXIF automatique :** Redressement natif des photos prises par des smartphones pour éviter que l'IA n'analyse une image à l'envers.
* **Data Flywheel (Auto-collecte) :** Chaque photo soumise au serveur est automatiquement classée et sauvegardée. Cela permet de créer une base de données d'images du "monde réel" pour ré-entraîner et améliorer continuellement l'intelligence artificielle.

---

## 📂 Structure du Dépôt

Pour fonctionner, ce projet s'articule autour des éléments suivants :

* `api.py` : Le script principal du serveur développé avec **FastAPI**. Il réceptionne les images, les prépare, lance l'IA, gère la sauvegarde et renvoie le "Top 3" des probabilités.
* `categories.txt` : La liste des catégories de matériaux reconnues par le modèle (utilisée pour traduire les résultats mathématiques en texte).
* `material_resnet_model.onnx` : Le modèle d'intelligence artificielle pré-entraîné *(assurez-vous de l'avoir dans votre dossier local)*.
* `dataset_collecte/` : Dossier généré automatiquement au premier lancement, où seront archivées les photos analysées pour le futur ré-entraînement.

---

## 🛠️ Installation et Démarrage

### 1. Prérequis
Assurez-vous d'avoir Python installé sur votre machine. Clonez ce dépôt sur votre PC :
```bash
git clone [https://github.com/votre-nom/material-scanner-api.git](https://github.com/votre-nom/material-scanner-api.git)
cd material-scanner-api


2. Installation des dépendances
Ouvrez votre terminal et installez les bibliothèques requises en une seule commande :



Bash
pip install fastapi uvicorn python-multipart onnxruntime numpy Pillow torch torchvision


3. Lancer le serveur local
Pour démarrer l'API, exécutez :



Bash
uvicorn api:app --host 0.0.0.0 --port 8000


Le serveur écoutera désormais sur le port 8000. Le paramètre 0.0.0.0 lui permet d'accepter les requêtes d'autres appareils sur votre réseau local (ex: votre téléphone).
🌍 Test à distance (Pour les testeurs)
Si vous souhaitez faire tester l'application mobile à des collaborateurs qui ne sont pas sur votre réseau Wi-Fi local, vous pouvez exposer ce serveur sur Internet très simplement grâce à ngrok :
Laissez tourner votre serveur uvicorn dans un premier terminal.
Téléchargez ngrok et ouvrez un second terminal.
Lancez la commande : ngrok http 8000
Copiez l'URL générée (ex: https://abcd-12-34.ngrok-free.app) et entrez-la dans les paramètres de l'application mobile de vos testeurs.
Développé pour l'innovation en ingénierie des matériaux
