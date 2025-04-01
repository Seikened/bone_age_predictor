import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from colorstreak import log
from tqdm import tqdm


#os.system('clear')

# --- Rutas ---
main_rute = os.get() + "/"
# --- Imgs rute ---
img_train_dataset = main_rute + 'boneage-training-dataset/'
img_test_dataset = main_rute + 'boneage-test-dataset/'
# --- CSV rute ---
csv_train = main_rute + 'boneage-training-dataset.csv'
csv_test = main_rute + 'boneage-test-dataset.csv'


df_train = pd.read_csv(csv_train)
df_test = pd.read_csv(csv_test)


# --- Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_img_from_path(path,id_img):
    rute_img = path + f"{id_img}.png"
    return rute_img



def cargar_datos():
    """Carga los CSV y retorna los dataframes de entrenamiento y test."""
    df_train = pd.read_csv(csv_train)
    df_test = pd.read_csv(csv_test)
    return df_train, df_test


def eda(df):
    """Realiza análisis exploratorio y visualiza la distribución de la variable 'boneage'."""
    log.debug(f"Train shape: {df.shape}")
    log.info("Primeras filas del dataset:")
    log.debug(df.head())
    log.info("="*50)

    log.info("Información del dataset:")
    log.debug(df.info())
    log.info("\nEstadísticas descriptivas:")
    log.debug(df.describe())
    
    log.info("\nValores faltantes:")
    log.debug(df.isnull().sum())
    
    # Visualización de la distribución de 'boneage'
    plt.figure(figsize=(10,6))
    sns.histplot(df['boneage'], bins=30, kde=True)
    plt.title("Distribución de la Edad Ósea (Training)")
    plt.xlabel("Edad Ósea")
    plt.ylabel("Frecuencia")
    plt.show()


def preparar_datos(df):
    """Separa features y target, divide en entrenamiento y validación, y aplica escalado."""
    X = df.drop('boneage', axis=1)
    y = df['boneage']
    
    # División 80-20
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    log.debug(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    # Escalado de features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled, y_train, y_val

def global_info():
    df_train, df_test = cargar_datos()
    eda(df_train)
    X_train_scaled, X_val_scaled, y_train, y_val = preparar_datos(df_train)
    log.debug(f"Datos preparados: X_train_scaled shape: {X_train_scaled.shape}")

    # Ejemplo de carga de imagen
    if 'image_path' in df_train.columns:
        sample_img = load_and_preprocess_image(df_train['image_path'].iloc[0])
        plt.imshow(sample_img)
        plt.title("Imagen de Ejemplo")
        plt.axis('off')
        plt.show()

# Opcional: función para procesamiento de imágenes si existe la columna 'image_path'
def load_and_preprocess_image(path, target_size=(224,224)):
    img = Image.open(path)
    img = img.resize(target_size)
    img = img.convert("RGB")
    return np.array(img)



# ========================== MODELOS ==========================
class BoneAgeImageDataset(Dataset):
    def __init__(self, df, images_folder, transform):
        """
        df: DataFrame que contiene al menos las columnas 'id' y 'boneage'
        images_folder: carpeta donde se encuentran las imágenes (ej. img_train_dataset)
        transform: transformaciones a aplicar (usaremos las del ViT preentrenado)
        """
        self.df = df.reset_index(drop=True)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construimos la ruta de la imagen usando la función que ya definiste
        img_path = get_img_from_path(self.images_folder, row['id'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Convertir la etiqueta a tensor (float) para regresión
        boneage = torch.tensor(row['boneage'], dtype=torch.float32)
        return image, boneage

# --- Función para entrenar (fine-tuning) el modelo ---
def fine_tuning_model(df, images_folder, num_epochs=20, batch_size=16, lr=1e-4):
    """
    Realiza el fine-tuning del modelo ViT_B_16 para predecir la edad ósea (boneage)
    usando el DataFrame 'df' (que proviene de csv_train) y las imágenes locales en 'images_folder'.
    """
    # Obtener transformaciones del modelo pre-entrenado (para mantener consistencia)
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    
    log.info("Iniciando separación de datos")
    # Dividir el dataframe en entrenamiento y validación (80-20)
    df_train_split, df_val_split = train_test_split(df, test_size=0.2, random_state=42)
    log.info(f"Train split: {df_train_split.shape}, Val split: {df_val_split.shape}")
    
    # Crear datasets y DataLoaders
    train_dataset = BoneAgeImageDataset(df_train_split, images_folder, transform)
    val_dataset   = BoneAgeImageDataset(df_val_split, images_folder, transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Crear el modelo ViT adaptado a regresión
    model = crear_modelo_vit_regresion()
    
    # Definir criterio de pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    log.info("Ciclo de entrenamiento")
    # Ciclo de entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Crear la barra de progreso dentro del ciclo de épocas
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)
        
        for i, (images, boneages) in enumerate(progress_bar):
            images = images.to(device)
            # Ajustamos boneages a forma (batch, 1)
            boneages = boneages.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, boneages)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Actualizamos la barra de progreso con el porcentaje y la pérdida
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.2f}",
                "Progress": f"{(i+1)*100/len(train_loader):.2f}%"
            })
        
        epoch_loss = running_loss / len(train_dataset)
        log.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}")

        
        # Evaluación en el conjunto de validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, boneages in val_loader:
                images = images.to(device)
                boneages = boneages.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, boneages)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_dataset)
        
        log.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return model

def crear_modelo_vit_regresion():
    """
    Carga el modelo ViT_B_16 con pesos pre-entrenados de ImageNet
    y reemplaza la capa final para regresión (salida 1).
    """
    # Usamos la versión estable IMAGENET1K_V1 (baseline robusto)
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    model = torchvision.models.vit_b_16(weights=weights)
    # Obtenemos el número de características de entrada de la cabeza original
    num_features = model.heads.head.in_features
    # Reemplazamos la cabeza por una capa lineal que produzca una única salida
    model.heads.head = nn.Linear(num_features, 1)
    model = model.to(device)
    return model

def predecir_boneage(model, image_path):
    """
    Dada la ruta a una imagen, la preprocesa y usa el modelo para predecir la edad ósea.
    """
    # Abrir la imagen y convertirla a RGB
    image = Image.open(image_path).convert("RGB")
    
    # Obtener los transforms asociados a los pesos (mantiene la consistencia)
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    
    # Preprocesar la imagen: resize, central crop, normalización, etc.
    input_tensor = preprocess(image)
    # Añadir dimensión de batch
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # La salida será un tensor de forma (1,1)
        output = model(input_batch)
    # Convertir la salida a un número (por ejemplo, float)
    return output.item()



def main():
    df_train, df_test = cargar_datos()
    #global_info()  # Para EDA y visualización
    
    # Definir ruta donde se guardarán los pesos del modelo
    model_weights_path = main_rute + 'vit_b16_finetuned.pth'
    
    if os.path.exists(model_weights_path):
        log.info("Cargando pesos guardados...")
        modelo_entrenado = crear_modelo_vit_regresion()
        modelo_entrenado.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        log.info("Iniciando fine-tuning del modelo ViT_B_16 para predecir boneage...")
        modelo_entrenado = fine_tuning_model(df_train, img_train_dataset, num_epochs=10, batch_size=16, lr=1e-4)
        torch.save(modelo_entrenado.state_dict(), model_weights_path)
        
        
    print(f"Valores del entrenamiento del rendimiento del modelo: {model_weights_path}")
    # Prueba rápida: predecir en una imagen de ejemplo
    id_mg =  1391
    ejemplo_img_path = get_img_from_path(img_train_dataset, df_train['id'].iloc[id_mg])
    prediccion = predecir_boneage(modelo_entrenado, ejemplo_img_path)
    print(f"Predicción de edad ósea para la imagen {df_train['id'].iloc[id_mg]}: {prediccion:.2f}")
    
    print(f"Valor real: {df_train['boneage'].iloc[id_mg]} | Predicción: {prediccion:.2f} | Error: {abs(df_train['boneage'].iloc[id_mg] - prediccion):.2f} ")
    
    
    


if __name__ == "__main__":
    main()
