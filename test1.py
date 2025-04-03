import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm  # Importar tqdm
import time  # Importar time para medir el tiempo de entrenamiento

# Configuración
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cpu")  # Forzar el uso de la CPU

# Definir ruta de los datos
path = "/home/sant/Documents/python/imss"
DATA_DIR = path  # Asegúrate de definir la ruta correctamente

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Cargar etiquetas
csv_path = os.path.join(DATA_DIR, "boneage-training-dataset.csv")
df = pd.read_csv(csv_path)

# Dataset personalizado
class BoneAgeDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.iloc[idx, 0]) + ".png"
        img_path = os.path.join(self.data_dir, "images", "boneage-training-dataset", "boneage-training-dataset", img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)
        return image, label

# Crear dataset y dataloader
dataset = BoneAgeDataset(df, DATA_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Cargar modelo ViT-B/16 preentrenado
model = models.vit_b_16(pretrained=True)
model.heads = torch.nn.Linear(model.heads.head.in_features, 1)  # Regresión
model.to(DEVICE)  # Asegúrate de mover el modelo a la CPU

# Configurar SWAG
swa_model = AveragedModel(model)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=LR)

# Entrenamiento
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Registrar el tiempo de inicio de la época
    epoch_start_time = time.time()

    # Barra de progreso para los batches dentro de la época
    with tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch') as bar:
        for images, labels in bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)  # Asegúrate de mover las imágenes y etiquetas a la CPU

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Actualizar la barra con el valor de pérdida
            bar.set_postfix(loss=running_loss/len(bar))

    # Calcular y mostrar el tiempo de la época
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")
    print(f"Epoch {epoch+1} took {epoch_duration:.2f} seconds.")

    if epoch >= EPOCHS - 5:
        swa_model.update_parameters(model)
        swa_scheduler.step()

# Actualizar las estadísticas de BN y guardar modelo
torch.optim.swa_utils.update_bn(dataloader, swa_model, DEVICE)
torch.save(swa_model.state_dict(), "/content/vit_bone_age_swag.pth")

print("Entrenamiento completado. Modelo SWAG guardado en /content/vit_bone_age_swag.pth")
