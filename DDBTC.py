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