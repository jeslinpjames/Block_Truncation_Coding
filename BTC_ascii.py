from bitarray import bitarray
import numpy as np
import cv2
import pickle

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
    
 
def save_encoded_data(encoded_data, mean_path,variance_path,blocks_path):
    if encoded_data:
        try:
            with open(mean_path, 'wb') as f:
                for mean in encoded_data['means']:
                    f.write(mean)
            with open(variance_path, 'wb') as f:
                for variance in encoded_data['variances']:
                    f.write(variance)
            with open(blocks_path, 'wb') as f:
                for block in encoded_data['quantized_data']:
                    block.tofile(f)
            print("Encoded data saved successfully")
        except Exception as e:
            print("Error in saving encoded data")
            print(e)

def to_char(value):
    if 0 <= value <= 255:
        return value.to_bytes(1, byteorder='big')
    else:
        raise ValueError("Value must be in the range 0-255")

def encode_BTC(img, block_size=4):
    if img is not None:
        height, width = img.shape
        mean_bytes = b""
        variance_bytes = b""
        encoded_data = {
            'block_size': block_size,
            'img_shape': img.shape,
            'means': [],
            'variances': [],
            'quantized_data': []
        }
        num=0
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img[i:i+block_size, j:j+block_size]
                mean = int(np.clip(np.mean(block), 0, 255))
                variance = int(np.clip(np.std(block), 0, 255))
                encoded_data['means'].append( to_char(mean))
                encoded_data['variances'].append( to_char(variance))
                binary_block = (block >= mean).astype(np.uint8)
                bit_array_block = bitarray(binary_block.flatten().tolist())
                encoded_data['quantized_data'].append(bit_array_block)
                num+=1
        print("Number of blocks: ",num)
    return encoded_data



if __name__=="__main__":
    img = load_image("D:/git/Block_Truncation_Coding/lena.png")
    if img is not None:
        print("Original Image Shape: ",img.shape)
        mat = np.array([
        [135, 42, 201, 173, 94, 117, 55, 208],
        [30, 183, 70, 150, 42, 88, 123, 77],
        [101, 162, 44, 95, 200, 35, 217, 124],
        [72, 56, 91, 13, 246, 180, 37, 64],
        [141, 232, 105, 168, 49, 87, 112, 19],
        [234, 99, 38, 78, 91, 221, 72, 53],
        [193, 11, 75, 63, 234, 150, 194, 87],
        [94, 201, 245, 168, 5, 113, 45, 142]
        ], dtype=np.uint8)
        # img = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        mean_output_path="D:/git/Block_Truncation_Coding/mean.txt"
        variance_output_path="D:/git/Block_Truncation_Coding/variance.txt"
        blocks_output_path="D:/git/Block_Truncation_Coding/blocks.txt"
        encoded_data= encode_BTC(img,block_size=4)
        save_encoded_data(encoded_data,mean_output_path,variance_output_path,blocks_output_path)