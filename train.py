import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torch.utils.data import WeightedRandomSampler
import os

NUM_CLASSES = 18
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),       # Comme dans tes tests manuels, c'est mieux !
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # Correction ici
])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Zoom aléatoire
    transforms.RandomHorizontalFlip(),                   # Effet miroir aléatoire
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Alterne la lumière et les couleurs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform_train)

# ==========================================
# CRÉATION DU SAMPLER PONDÉRÉ
# ==========================================
print("Analyse des fichiers pour l'échantillonnage pondéré...")

POIDS_NORMAL = 1.0  # Pour MINC et Kaggle
POIDS_PERSO = 30.0  # Pour tes photos (Mégaphone x50)

poids_des_images = []
compteur_perso = 0

# train_dataset.samples contient des tuples : (chemin_complet, index_de_la_classe)
for chemin_image, _ in train_dataset.samples:
    nom_fichier = os.path.basename(chemin_image).lower()
    
    # Règle de détection : adapte "img", "pxl" ou "dsc" selon ce que produit ton téléphone
    # Si tu as renommé tes photos avec "perso_", ajoute-le dans le tuple !
    if nom_fichier.startswith(('skin_2026', 'stone_2026', 'sky_2026', 'wood_2026','metal_2026','carpet_2026',
                               'fabric_2026','painted_2026','glass_2026','foliage_2026','ceramic_2026','leather_2026',
                               'paper_2026','tile_2026','plastic_2026')):
        poids_des_images.append(POIDS_PERSO)
        compteur_perso += 1
    else:
        poids_des_images.append(POIDS_NORMAL)

print(f"-> {compteur_perso} photos personnelles identifiées et boostées.")

# Convertir la liste en tenseur PyTorch
weights_tensor = torch.DoubleTensor(poids_des_images)

# Créer le Sampler
sampler_pondere = WeightedRandomSampler(
    weights=weights_tensor,
    num_samples=len(weights_tensor), # On garde la même taille d'Epoch qu'avant
    replacement=True # CRUCIAL : Permet de piocher la même photo perso plusieurs fois !
)

# ==========================================
# MISE À JOUR DU DATALOADER
# ==========================================
# ATTENTION : Il faut obligatoirement enlever 'shuffle=True' quand on utilise un Sampler,
# car c'est le Sampler qui s'occupe désormais du mélange et du tirage au sort.
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler_pondere 
)

# La suite de ton code reste identique (val_dataset, etc.)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print(f"Class detected {class_names}")


# ==========================================
# 1. INITIALISATION DU NOUVEAU MODÈLE (ResNet50)
# ==========================================
print("Chargement de ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# ==========================================
# 2. PRÉPARATION DU VRAI TRANSFER LEARNING
# ==========================================
# On gèle toutes les couches de base pour protéger la "mémoire visuelle" du modèle
for param in model.parameters():
    param.requires_grad = False

# On remplace la dernière couche (sur ResNet, elle s'appelle 'fc' pour Fully Connected)
# Par défaut, elle est dégelée car on vient de la créer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ==========================================
# FONCTION D'ENTRAÎNEMENT (Pour éviter de répéter le code)
# ==========================================
def train_model(epochs, optimizer, phase_name, scheduler=None):
    print(f"\n=== DÉMARRAGE {phase_name} ({epochs} Epochs) ===")
    
    best_val_loss = float('inf') # On initialise le record à l'infini
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            print(f"   ⭐ Nouveau record ! Val Loss baisse de {best_val_loss:.4f} à {val_loss:.4f}")
            best_val_loss = val_loss
            # On copie le cerveau du modèle à cet instant précis
            best_model_weights = copy.deepcopy(model.state_dict())

        # --- ACTIVATION DU SCHEDULER ---
        if scheduler is not None:
                    # On donne la note d'examen au scheduler pour qu'il juge s'il doit agir
                    scheduler.step(val_loss)
                    
                    # Affichage du Learning Rate actuel pour voir quand il baisse
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"   -> Learning Rate : {current_lr:.6f}")

            # LIGNES DÉPLACÉES ICI (En dehors de la boucle 'for', mais dans la fonction 'def')
        print(f"Fin de la {phase_name}. Restauration du meilleur modèle (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_weights)
    

# ==========================================
# PHASE 1 : Entraînement de la "tête" uniquement
# ==========================================
# Seule la dernière couche apprend. Le reste est bloqué.
# On utilise un learning rate standard (0.001)
optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=0.001)
train_model(epochs=10, optimizer=optimizer_phase1, phase_name="PHASE 1 (Tête uniquement)")

# ==========================================
# PHASE 2 : Affinage global (Fine-Tuning)
# ==========================================
for param in model.parameters():
    param.requires_grad = True

optimizer_phase2 = optim.Adam(model.parameters(), lr=0.0001) 

# CRÉATION DU SCHEDULER
# mode='min' : On surveille une valeur qui doit baisser (le Val Loss)
# factor=0.1 : Si ça stagne, on multiplie le LR par 0.1 (on le divise par 10)
# patience=2 : On attend 2 epochs sans amélioration avant de sévir
scheduler_phase2 = ReduceLROnPlateau(optimizer_phase2, mode='min', factor=0.1, patience=2)

train_model(
    epochs=15, 
    optimizer=optimizer_phase2, 
    phase_name="PHASE 2 (Affinage global avec Scheduler)", 
    scheduler=scheduler_phase2
)

# ==========================================
# EXPORTATION EN ONNX
# ==========================================
print("\nExportation du modèle en ONNX...")
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
torch.onnx.export(
    model,
    dummy_input,
    "material_resnet_model.onnx", 
    export_params=True,
    input_names=['input'],
    output_names=['output']
)

print("Modèle exporté sous le nom 'material_resnet_model.onnx' !")

torch.save(model.state_dict(), "material_resnet_model.pth")
print("Modèle natif PyTorch sauvegardé 'material_resnet_model.pth' !")