import os 
import shutil

TXT_FOLDER = "minc-2500/labels"

IMAGE_FOLDER = "minc-2500/images"

OUTPUT_FOLDER = "dataset"

USED_MATS = []

with open('minc-2500/categories.txt') as file:
    for line in file:
        USED_MATS.append(line.strip())

def prepare_split(txt_filename,split_name):
    txt_path = os.path.join(TXT_FOLDER,txt_filename)

    if not os.path.exists(txt_path):
        print(f"Error : Path {txt_path} don't exist")
        return 
    
    with open(txt_path,'r') as f:
        lines = f.readlines()

    compteur = 0

    for line in lines:
        line = line.strip()
        if not line: continue

        parts = line.split('/')
        if len(parts) < 3:continue
        classe = parts[1]
        filename = parts[2]

        if classe in USED_MATS:
            dest_dir = os.path.join(OUTPUT_FOLDER,split_name,classe)
            os.makedirs(dest_dir,exist_ok=True)

            src_path = os.path.join(IMAGE_FOLDER,classe,filename)
            dest_path = os.path.join(dest_dir,filename)

            if os.path.exists(src_path) and not os.path.exists(dest_path):
                shutil.copy(src_path,dest_path)
                compteur += 1

    print(f"[{split_name}] {compteur} images copiées depuis {txt_filename}.")

print("Dataset Creation")

prepare_split("train1.txt","train")
prepare_split("validate1.txt","val")

print("Finished")

    