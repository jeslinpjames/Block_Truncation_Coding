import cv2
from bitarray import bitarray
import numpy as np

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

def save_image(img,path):        
    if img is not None:
        cv2.imwrite(path,img)
    else:
        print("Image not found")

def abs_moment(matrix, mean, m):
    return np.sum(np.abs(matrix - mean)) / m

def to_char(value):
    if 0 <= value <= 255:
        return value.to_bytes(1, byteorder='big')
    else:
        raise ValueError("Value must be in the range 0-255")

def encode_AMBTC(img, block_size=4):
    if img is not None:
        height, width = img.shape
        encoded_data = {
            'block_size': block_size,
            'img_shape': img.shape,
            'means': [],
            'abs_moment': [],
            'quantized_data': []
        }
        num=0
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img[i:i+block_size, j:j+block_size]
                mean = int(np.clip(np.mean(block), 0, 255))
                abs_moment = abs_moment(block,mean,block_size*block_size)
                encoded_data['means'].append( to_char(mean))
                encoded_data['abs_moment'].append( to_char(abs_moment))
                binary_block = (block >= mean).astype(np.uint8)
                bit_array_block = bitarray(binary_block.flatten().tolist())
                encoded_data['quantized_data'].append(bit_array_block)
                num+=1
        print("Number of blocks: ",num)
    return encoded_data

if __name__ =="__main__":
    mat = np.array([
        [121,114,56,47],
        [37,200,247,255],
        [16,0,12,169],
        [43,5,7,251]
    ], dtype=np.uint8)
    mean = int(np.clip(np.mean(mat), 0, 255))
    mean = np.mean(mat)
    abs_moment = abs_moment(mat,mean,4*4)
    print("Mean: ",mean)
    print("Abs moment: ",abs_moment)
