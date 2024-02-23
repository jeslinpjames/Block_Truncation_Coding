import numpy as np
from skimage.util import view_as_blocks
import cv2
from bitarray import bitarray
from PIL import Image

def load_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        else:
            print("Image not found")
            return None
    except Exception as e:
        print("Error in loading image")
        print(e)
        return None
    
def display_image(img):
    if img is not None:
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found")

def save_image(img, path):
    if img is not None:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.convert('L')
        pil_img.save(path, 'BMP', bits=8)
    else:
        print("Image not found")

def to_char(value):
    if 0 <= value <= 255:
        return value.to_bytes(1, byteorder='big')
    else:
        raise ValueError("Value must be in the range 0-255")


def load_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        else:
            print("Image not found")
            return None
    except Exception as e:
        print("Error in loading image")
        print(e)
        return None
def default_class_matrix(size=(8,8)):
    return np.array([[ 63 , 55 , 47 , 39 , 31 , 23 , 15 , 7],  
                     [ 62 , 54 , 46 , 38 , 30 , 22 , 14 , 6],
                     [ 61 , 53 , 45 , 37 , 29 , 21 , 13 , 5],
                     [ 60 , 52 , 44 , 36 , 28 , 20 , 12 , 4],
                     [ 59 , 51 , 43 , 35 , 27 , 19 , 11 , 3],
                     [ 58 , 50 , 42 , 34 , 26 , 18 , 10 , 2],
                     [ 57 , 49 , 41 , 33 , 25 , 17 , 9  , 1],
                     [ 56 , 48 , 40 , 32 , 24 , 16 , 8  , 0]])  

def floyd_steinberg_diff_matrix():
    return np.array([[0, 0, 0, 7/16, 1/16],  
                     [3/16, 5/16, 1/16, 0, 0],
                     [0, 0, 0, 0, 0]])


def diffuse(block, class_matrix, diff_matrix, xmin, xmax):
    block = block.copy()
    error = np.zeros_like(block)
    
    for i in range(class_matrix.shape[0]):
        for j in range(class_matrix.shape[1]):
            if class_matrix[i,j] != 0:
                continue
                
            x = block[i,j] + error[i,j]
            
            if x > (xmin + xmax) / 2.:
                y = float(xmax)  
            else:
                y = float(xmin)
                
            e = float(x) - y
            
            sum_weights = np.sum(diff_matrix[k,l] for k in range(diff_matrix.shape[0]) 
                                                for l in range(diff_matrix.shape[1]) 
                                                if class_matrix[k,l] > class_matrix[i,j])
            
            for k in range(diff_matrix.shape[0]):
                for l in range(diff_matrix.shape[1]):
                    new_i = i + k - 1
                    new_j = j + l - 1
                    if 0 <= new_i < error.shape[0] and 0 <= new_j < error.shape[1]:
                        error[new_i, new_j] += (e * diff_matrix[k,l])/sum_weights
                
            block[i,j] = y
            
    return block, error



def encode_DDBTC(img, block_size=(8,8), class_matrix=None, diff_matrix=None):
    if img is not None:
        height, width = img.shape
        encoded_data = {
            'block_size': block_size,
            'img_shape': img.shape,
            'means': [],
            'variances': [],
            'quantized_data': [],
            'maximums': [],
            'minimums': [],
        }

        if class_matrix is None:
            class_matrix = default_class_matrix()
        if diff_matrix is None:
            diff_matrix = floyd_steinberg_diff_matrix()

        bitmap = np.zeros(img.shape)
        out = np.zeros(img.shape)

        for i in range(0, height, block_size[0]):
            for j in range(0, width, block_size[1]):
                block = img[i:i+block_size[0], j:j+block_size[1]]
                xmin = block.min()
                xmax = block.max()
                block, error = diffuse(block, class_matrix, diff_matrix, xmin, xmax)
                bitmap[i:i+block_size[0], j:j+block_size[1]] = (block == xmax)
                out[i:i+block_size[0], j:j+block_size[1]] = xmin * (1 - bitmap[i:i+block_size[0], j:j+block_size[1]])
                out[i:i+block_size[0], j:j+block_size[1]] += xmax * bitmap[i:i+block_size[0], j:j+block_size[1]]
                encoded_data['minimums'].append(to_char(int(xmin)))
                encoded_data['maximums'].append(to_char(int(xmax)))

                # Compute and store the mean and variance for each block
                mean = int(np.clip(np.mean(block),  0,  255))
                variance = int(np.clip(np.std(block),  0,  255))
                encoded_data['means'].append(to_char(mean))
                encoded_data['variances'].append(to_char(variance))

                # Quantize the block and store it
                binary_block = (block >= mean).astype(np.uint8)
                bit_array_block = bitarray(binary_block.flatten().tolist())
                encoded_data['quantized_data'].append(bit_array_block)

        return encoded_data
    else:
        print("Image not found")
        return None


def reconstruct_DDBTC(encoded_data):
    if encoded_data:
        block_size = encoded_data['block_size']
        img_height, img_width = encoded_data['img_shape']
        means = encoded_data['means']
        variances = encoded_data['variances']
        quantized_data = encoded_data['quantized_data']
        reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
        block_id =  0
        
        for i in range(0, img_height, block_size[0]):
            for j in range(0, img_width, block_size[1]):
                bit_array_block = quantized_data[block_id]
                numpy_array = np.array(bit_array_block.tolist(), dtype=np.uint8)
                binary_block = numpy_array.reshape((block_size[0], block_size[1]))
                mean = int.from_bytes(means[block_id], byteorder='big')
                variance = int.from_bytes(variances[block_id], byteorder='big')
                xmin = int.from_bytes(encoded_data['minimums'][block_id], byteorder='big')
                xmax = int.from_bytes(encoded_data['maximums'][block_id], byteorder='big')
                reconstructed_block = np.where(binary_block ==  1, xmax, xmin)
                reconstructed_image[i:i + block_size[0], j:j + block_size[1]] = reconstructed_block
                block_id +=  1

        return reconstructed_image


img_path = "images/synthetic.png"
img = load_image(img_path)
out= encode_DDBTC(img)
reconstructed_image=reconstruct_DDBTC(out)
save_image(reconstructed_image, "D:/git/Block_Truncation_Coding/images/compressedDDBTC_img.bmp")
