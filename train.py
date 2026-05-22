import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models

NUM_CLASSES = 23
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.229,0.225])
])

train_dataset = datasets.ImageFolder(root='dataset/train',transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

class_names = train_dataset.classes
print(f"Class detected {class_names}")


model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

model.classifier[1] = nn.Linear(model.last_channel,NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


# Training

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.5

    for image,labels in train_loader:
        images,labels = image.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")


# Exportation to ONNX

model.eval()
dummy_input = torch.randn(1,3,224,224).to(DEVICE)
torch.onnx.export(
    model,
    dummy_input,
    "material_model.onnx",
    export_params=True,
    input_names=['input'],
    output_names=['output']
)

print("Model exported 'material_model.onnx'")