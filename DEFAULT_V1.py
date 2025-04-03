import os
import time

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

import warnings

import pandas as pd
import torch
import torchvision.models as models  # noqa: F401
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16
from tqdm import tqdm


def main():
    print(
        str(torch.cuda.is_available()) + ",",
        str(torch.cuda.current_device()) + ",",
        str(torch.cuda.get_device_name(torch.cuda.current_device())),
    )

    # Configuración
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-4
    DEVICE = torch.device("cuda")

    # Definir ruta de los datos (ajustado para acceso local)
    DATA_DIR = "C:/Users/Santiago/Documents/imss/images"  # Reemplaza con la ruta donde tienes el dataset descargado

    # Transformaciones de imagen
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

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
            img_path = os.path.join(self.data_dir, "boneage-training-dataset", img_name)
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            label = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)
            return image, label

    # Crear dataset y dataloader
    dataset = BoneAgeDataset(df, DATA_DIR, transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )  # Establecer num_workers=0 para evitar problemas en Windows

    # Cargar modelo ViT-B/16 con DEFAULT
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads = torch.nn.Linear(model.heads.head.in_features, 1)  # Regresión
    model.to(DEVICE)

    # Configurar SWAG
    swa_model = AveragedModel(model)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    swa_scheduler = SWALR(
        optimizer, anneal_strategy="linear", anneal_epochs=10, swa_lr=LR
    )

    # Entrenamiento con tqdm para barras de progreso
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=True,
            unit="batch",
            mininterval=2,
        ):
            images, labels = (
                images.to(DEVICE, non_blocking=True),
                labels.to(DEVICE, non_blocking=True).unsqueeze(1),
            )

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        print(
            f"Epoch {epoch + 1} completed in {time_str} (hh:mm:ss) - Loss: {running_loss / len(dataloader):.4f}\n"
        )

        if epoch >= EPOCHS - 10:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    # Actualizar las estadísticas de BN y guardar modelo
    torch.optim.swa_utils.update_bn(dataloader, swa_model, DEVICE)
    torch.save(swa_model.state_dict(), "vit_bone_age_default.pth")

    print(
        "Entrenamiento completado. Modelo SWAG LINEAR guardado en vit_bone_age_default.pth"
    )

    # Verificación del modelo con un batch de prueba
    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(dataloader))
        test_images, test_labels = test_images.to(DEVICE), test_labels.to(DEVICE)
        predictions = model(test_images).squeeze(1)

        for i in range(min(5, len(test_images))):  # Mostrar 5 ejemplos
            real_age = test_labels[i].item()
            predicted_age = predictions[i].item()
            error = abs(real_age - predicted_age)
            print(
                f"Real: {real_age:.2f} meses, Predicho: {predicted_age:.2f} meses, Error: {error:.2f} meses"
            )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")
    main()
