import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from colorstreak import log



class Keypoint:
    def __init__(self, x, y, label=None):
        """
        Representa un punto anatómico en coordenadas (x, y).
        label se asignará después al cluster correspondiente.
        """
        self.x = x
        self.y = y
        self.label = label

class Finger:
    def __init__(self, name, keypoints):
        """
        name es el nombre del dedo ("thumb", "index", ...)
        keypoints es la lista de puntos ordenados a lo largo del hueso.
        """
        self.name = name
        self.keypoints = keypoints

class Hand:
    def __init__(self, fingers):
        """
        fingers es una lista de instancias Finger,
        representando cada uno de los cinco dedos.
        """
        self.fingers = fingers

def load_image(path):
    # **Carga** la imagen en escala de grises
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def preprocess(img):
    # **Equalización adaptativa** para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def segment_bones(img, low=50, high=150):
    # **Segmentación** por Canny + dilatación para cerrar huecos
    edges = cv2.Canny(img, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    return cv2.dilate(edges, kernel)

def detect_rods(mask, threshold=50, min_line_length=30, max_line_gap=5):
    # **Detección de varillas** mediante HoughLinesP
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=threshold,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines.reshape(-1,4) if lines is not None else []

def extract_endpoints(lines):
    # **Extracción de extremos** de cada línea como posibles epífisis
    points = []
    for x1,y1,x2,y2 in lines:
        points.append(Keypoint(x1, y1))
        points.append(Keypoint(x2, y2))
    return points

def cluster_by_finger(points, n_clusters=5):
    # **Agrupamiento** de puntos en 5 clusters (dedos)
    coords = np.array([[p.x, p.y] for p in points])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    clusters = {i: [] for i in range(n_clusters)}
    for label, p in zip(kmeans.labels_, points):
        clusters[label].append(p)
    names = ['thumb','index','middle','ring','pinky']
    return {names[i]: clusters[i] for i in clusters}

def sort_along_bone_axis(pts):
    # **Orden** simple de puntos por coordenada y (de proximal a distal)
    return sorted(pts, key=lambda p: p.y)

def detect_hand(path):
    log.debug('Detectando mano en la imagen')
    img = load_image(path)
    
    log.debug('Imagen cargada y preprocesada')
    pre = preprocess(img)
    
    log.debug('Preprocesamiento realizado')
    mask = segment_bones(pre)
    
    log.debug('Segmentación de huesos realizada')
    lines = detect_rods(mask)
    
    log.debug('Detección de varillas realizada')
    points = extract_endpoints(lines)
    
    log.debug('Extracción de extremos realizada')
    clusters = cluster_by_finger(points)
    
    log.debug('Agrupamiento por dedos realizado')
    fingers = []
    
    for name, pts in clusters.items():
        finger = Finger(name, sort_along_bone_axis(pts))
        fingers.append(finger)
    log.debug(f'Dedo {name} detectado con {len(pts)} puntos')
    return Hand(fingers)


ruta_data_set = os.getcwd() + '/boneage-training-dataset/' 

# ID img (1377-15610)
id_img = 1377
log.debug(f'ID de imagen: {id_img}')
ruta_data_set = os.path.join(ruta_data_set, str(id_img) + '.png')

hand = detect_hand(ruta_data_set)
