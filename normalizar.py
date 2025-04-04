import glob
import os
from multiprocessing import Pool

import cv2
from tqdm import tqdm


def procesar_imagen(args):
    img_path, output_folder, umbral = args

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"Imagen no encontrada: {img_path}"

    _, img_bin = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)

    nombre = os.path.basename(img_path)
    output_path = os.path.join(output_folder, nombre)
    cv2.imwrite(output_path, img_bin)
    return


def main(input_folder, output_folder, umbral=128):
    os.makedirs(output_folder, exist_ok=True)
    img_paths = glob.glob(os.path.join(input_folder, "*.*"))

    args_list = [(path, output_folder, umbral) for path in img_paths]

    num_processes = os.cpu_count()

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap(procesar_imagen, args_list), total=len(args_list))
        )


if __name__ == "__main__":
    input_folder = r"images\db_imss"
    output_folder = r"images\db_imss_normalized"
    main(input_folder, output_folder, umbral=114)
