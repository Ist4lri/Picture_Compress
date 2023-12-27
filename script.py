import sys
import numpy as np
from scipy.fft import dct, idct
from PIL import Image

def blockify(image):
    h, w, _ = image.shape
    bh, bw = 8, 8
    return [image[i:i+bh, j:j+bw, :] for i in range(0, h, bh) for j in range(0, w, bw)]

def unblockify(blocks, shape):
    h, w, c = shape
    bh, bw = 8, 8
    image = np.zeros(shape)
    for i, block in enumerate(blocks):
        row = (i // (w // bw)) * bh
        col = (i % (w // bw)) * bw
        image[row:row+bh, col:col+bw, :] = block
    return image

def jpeg_compress(image_path, output_path, quality=50):
    # Charger l'image en couleur
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)

    # Étape 1: Division de l'image en blocs de 8x8 pour chaque canal de couleur
    blocks = blockify(img_array)

    # Étape 2: Transformation en cosinus discrète (DCT) pour chaque canal de couleur
    dct_blocks = [dct(block, norm='ortho', axis=0) for block in blocks]

    # Étape 3: Quantification pour chaque canal de couleur
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                          [12, 12, 14, 19, 26, 58, 60, 55],
                                          [14, 13, 16, 24, 40, 57, 69, 56],
                                          [14, 17, 22, 29, 51, 87, 80, 62],
                                          [18, 22, 37, 56, 68, 109, 103, 77],
                                          [24, 35, 55, 64, 81, 104, 113, 92],
                                          [49, 64, 78, 87, 103, 121, 120, 101],
                                          [72, 92, 95, 98, 112, 100, 103, 99]])

    quantized_blocks = [np.round(dct_block / quantization_matrix[..., None] * quality) for dct_block in dct_blocks]

    # Étape 4: Sauvegarde de l'image compressée
    compressed_blocks = [idct(quantized_block, norm='ortho', axis=0) for quantized_block in quantized_blocks]
    compressed_image = unblockify(compressed_blocks, img_array.shape)
    compressed_image[compressed_image < 0] = 0
    compressed_image[compressed_image > 255] = 255

    # Correction de la luminosité
    compressed_image *= 255.0 / compressed_image.max()

    compressed_img = Image.fromarray(compressed_image.astype(np.uint8))
    compressed_img.save(output_path)

if __name__ == "__main__":
    # Vérifier le nombre d'arguments
    if len(sys.argv) != 4:
        print("Usage: python3 script.py input_image_path output_image_path quality")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        quality = int(sys.argv[3]) if len(sys.argv) == 4 else 50
        jpeg_compress(input_image_path, output_image_path, quality)
