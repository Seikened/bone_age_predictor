"""
para utilizarlo, en la consola escribe:
python CGPT_TEST_SWAG_LINEAR_V1.py --folder <ubicacion-del-folder-con-fotos> --n <cantidad-de-fotos-para-probar>
"""

import argparse
import os
import random

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16


def load_model(model_path, device):
    from collections import OrderedDict

    # Cargar el modelo base con los pesos SWAG_LINEAR_V1
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    model = vit_b_16(weights=weights)
    # Ajuste para regresión: salida de una neurona
    model.heads = torch.nn.Linear(model.heads.head.in_features, 1)

    # Cargar el state dict
    state_dict = torch.load(model_path, map_location=device)

    # Remover el prefijo "module." de todas las llaves
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    # Cargar el state dict con strict=False para omitir llaves inesperadas
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def process_images(folder, n, model, device, transform, csv_path=None):
    # Listar archivos de imagen (asumimos .png)
    image_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    if not image_files:
        print("No se encontraron imágenes en la carpeta:", folder)
        return

    # Seleccionar n imágenes aleatoriamente (o todas si son menos)
    if len(image_files) > n:
        image_files = random.sample(image_files, n)
    else:
        print("La cantidad de imágenes es menor que n, se procesarán todas.")

    errors = []
    predictions = {}

    # Si se proporciona CSV, cargar datos en un diccionario {id: boneage}
    csv_data = {}
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            csv_data[str(row["id"])] = row["boneage"]

    for img_file in image_files:
        img_path = os.path.join(folder, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("Error al abrir la imagen", img_path, ":", e)
            continue

        # Aplicar transformaciones y agregar dimensión de batch
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor).item()
        predictions[img_file] = pred

        # Si hay CSV, buscar el valor real y calcular error
        if csv_path is not None:
            img_id = os.path.splitext(img_file)[0]
            if img_id in csv_data:
                true_val = csv_data[img_id]
                error = abs(true_val - pred)
                errors.append(error)
                print(
                    f"Imagen: {img_file} | Real: {true_val} | Predicho: {pred:.2f} | Error: {error:.2f}"
                )
            else:
                print(f"Imagen: {img_file} | Sin datos reales en el CSV")
        else:
            print(f"Imagen: {img_file} | Predicción: {pred:.2f}")

    if errors:
        avg_error = sum(errors) / len(errors)
        print(f"\n**Promedio de error**: {avg_error:.2f} meses")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Script para predecir edad ósea usando ViT fine tuned"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Carpeta de imágenes ('boneage-training-dataset' o 'subtest')",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="vit_bone_age_swag.pth",
        help="Ruta al archivo .pth del modelo",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Número de imágenes a procesar"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # Definir transformaciones: redimensionar, tensor y normalización
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Cargar modelo
    model = load_model(args.model_path, device)

    # Determinar si se usará CSV (solo para 'boneage-training-dataset')
    csv_path = None
    folder_name = os.path.basename(os.path.normpath(args.folder))
    if folder_name == "boneage-training-dataset":
        # Se asume que el CSV está en el directorio padre de la carpeta de imágenes
        parent_dir = os.path.dirname(os.path.normpath(args.folder))
        csv_candidate = os.path.join(parent_dir, "boneage-training-dataset.csv")
        if os.path.exists(csv_candidate):
            csv_path = csv_candidate
        else:
            print("No se encontró el CSV en:", csv_candidate)

    # Procesar imágenes y mostrar predicciones
    process_images(args.folder, args.n, model, device, transform, csv_path)


if __name__ == "__main__":
    main()
