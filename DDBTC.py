import numpy as np
import cv2
from bitarray import bitarray
from PIL import Image
from check_psnr import calculate_psnr

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
    return np.array([[35, 28,  0, 47, 34, 32, 37, 21],
                   [10, 59, 18, 60, 41, 27, 12, 33],
                   [58, 36,  8, 25,  6, 49, 43, 26],
                   [16, 29,  2, 44, 38, 15, 19, 56],
                   [17, 42, 20, 50, 55, 24, 30, 13],
                   [39, 14,  7, 61, 22,  1, 57, 54],
                   [51, 31, 45,  9, 40, 11, 53,  4],
                   [23, 62, 63, 52,  5, 48, 46,  3]]
                  )

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
                mean = np.mean(block)
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
        quantized_data = encoded_data['quantized_data']
        reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
        block_id =  0
        
        for i in range(0, img_height, block_size[0]):
            for j in range(0, img_width, block_size[1]):
                bit_array_block = quantized_data[block_id]
                numpy_array = np.array(bit_array_block.tolist(), dtype=np.uint8)
                binary_block = numpy_array.reshape((block_size[0], block_size[1]))
                xmin = int.from_bytes(encoded_data['minimums'][block_id], byteorder='big')
                xmax = int.from_bytes(encoded_data['maximums'][block_id], byteorder='big')
                reconstructed_block = np.where(binary_block ==  1, xmax, xmin)
                reconstructed_image[i:i + block_size[0], j:j + block_size[1]] = reconstructed_block
                block_id +=  1

        return reconstructed_image
    
if __name__ =="__main__":
    img = load_image("images/synthetic.bmp")
    if img is not None:
        print("Original Image Shape: ",img.shape)
        mat = np.array([
            [121,114,56,47],
            [37,200,247,255],
            [16,0,12,169],
            [43,5,7,251]
        ], dtype=np.uint8)
        max_output_path="compressed/max.txt"
        min_output_path="compressed/min.txt"
        blocks_output_path="compressed/blocks.txt"
        encoded_data= encode_DDBTC(img)
        # save_encoded_data(encoded_data,max_output_path,min_output_path,blocks_output_path)
        # encoded_data=load_encoded_data(max_output_path,min_output_path,blocks_output_path,img,block_size=(8,8))
        reconstructed_image=reconstruct_DDBTC(encoded_data)
        save_image(reconstructed_image, "images/compressedDDBTC_img.bmp")
        output_path = "images/synthetic.bmp"
        path2="images/compressedDDBTC_img.bmp"
        psnr_value = calculate_psnr(output_path, path2)
        print(f"PSNR for original image and compressed image:  {psnr_value:.2f}")