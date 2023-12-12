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

def find_abs_moment(matrix, mean, m):
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
                m = block_size * block_size
                mean = np.mean(block)
                abs_moment = find_abs_moment(block,mean,m)
                abs_moment = int(np.clip(abs_moment, 0, 255))
                mean = int(np.clip(mean, 0, 255))
                encoded_data['means'].append( to_char(mean))
                encoded_data['abs_moment'].append( to_char(abs_moment))
                binary_block = (block >= mean).astype(np.uint8)
                bit_array_block = bitarray(binary_block.flatten().tolist())
                encoded_data['quantized_data'].append(bit_array_block)
                num+=1
        print("Number of blocks: ",num)
    return encoded_data

def reconstruct_BTC(encoded_data):
    if encoded_data:
        block_size = encoded_data['block_size']
        img_height, img_width = encoded_data['img_shape']
        means = [int.from_bytes(mean, byteorder='big') for mean in encoded_data['means']]
        abs_moments = [int.from_bytes(abs_moment, byteorder='big') for abs_moment in encoded_data['abs_moment']]
        quantized_data = encoded_data['quantized_data']
        reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
        block_id = 0
            
        for i in range(0, img_height, block_size):
            for j in range(0, img_width, block_size):
                bit_array_block = quantized_data[block_id]
                numpy_array = np.array(bit_array_block.tolist(), dtype=np.uint8)
                binary_block = numpy_array.reshape((4, 4))
                q = np.sum(binary_block)
                mean = means[block_id]
                abs_moment = abs_moments[block_id]
                m = block_size * block_size
                q = np.sum(binary_block)
                m = block_size * block_size
                gamma = m * abs_moment / 2
                if q != 0 and q != block_size**2:
                    # Compute 'a' and 'b'
                    b = mean + (gamma / q)
                    a = mean - (gamma / (m - q))
                else:
                    # Compute 'a' and 'b'
                    a = int(mean - gamma)
                    b = int(mean + gamma)
                # Use 'a' and 'b' to reconstruct the block
                binary_block = binary_block.reshape((block_size, block_size))
                reconstructed_block = np.zeros((block_size, block_size), dtype=np.uint8)
                for k in range(block_size):
                    for l in range(block_size):
                        if binary_block[k, l] == 1:  
                            reconstructed_block[k, l] = b
                        else:
                            reconstructed_block[k, l] = a
                reconstructed_image[i:i + block_size, j:j + block_size] = reconstructed_block
                block_id += 1
    return reconstructed_image

if __name__ =="__main__":
    mat = np.array([
        [121,114,56,47],
        [37,200,247,255],
        [16,0,12,169],
        [43,5,7,251]
    ], dtype=np.uint8)
    mean = int(np.clip(np.mean(mat), 0, 255))
    mean = np.mean(mat)
    abs_moment = find_abs_moment(mat,mean,4*4)
    gamma = 16*abs_moment/2
    
    encoded =encode_AMBTC(mat)
    img = reconstruct_BTC(encoded)
    print(img)

