import cv2
from bitarray import bitarray
import numpy as np
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

def save_image(img,path):        
    if img is not None:
        cv2.imwrite(path,img)
    else:
        print("Image not found")

def to_char(value):
    if 0 <= value <= 255:
        return value.to_bytes(1, byteorder='big')
    else:
        raise ValueError("Value must be in the range 0-255")
    
def save_encoded_data(encoded_data, uh_path,ul_path,blocks_path):
    if encoded_data:
        try:
            with open(uh_path, 'wb') as f:
                for uh in encoded_data['U_high']:
                    f.write(uh)
            with open(ul_path, 'wb') as f:
                for ul in encoded_data['U_low']:
                    f.write(ul)
            with open(blocks_path, 'wb') as f:
                for block in encoded_data['quantized_data']:
                    block.tofile(f)
            print("Encoded data saved successfully")
        except Exception as e:
            print("Error in saving encoded data")
            print(e)

def load_encoded_data(uh_path, ul_path, blocks_path, block_size=4 ):
    try:
        encoded_data = {
            'block_size': block_size,
            'img_shape': None,
            'U_high': [],
            'U_low': [],
            'quantized_data': []
        }
        with open(uh_path, 'rb') as f:
            encoded_data['U_high'] = list(f.read())
        with open(ul_path, 'rb') as f:
            encoded_data['U_low'] = list(f.read())
        with open(blocks_path, 'rb') as f:
            # Read bytes from the file
            size = block_size * block_size
            size = size // 8 
            byte_block = f.read(size)     
            while byte_block:
                # Convert bytes to bitarray
                bit_array_block = bitarray()
                bit_array_block.frombytes(byte_block) 
                # Append the bitarray block to the list
                encoded_data['quantized_data'].append(bit_array_block)
                # Read the next block
                byte_block = f.read(size)
        print("Encoded data loaded successfully")
        return encoded_data
    except Exception as e:
        print("Error in loading encoded data")
        print(e)
        return None

def encode_MBTC(img, block_size=4):
    if img is not None:
        height, width = img.shape
        encoded_data = {
            'block_size': block_size,
            'img_shape': img.shape,
            'U_high': [],
            'U_low': [],
            'quantized_data': []
        }
        num=0
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img[i:i+block_size, j:j+block_size]
                mean = int(np.clip(np.mean(block), 0, 255))
                T = 0
                if mean == 0:
                    uh = 0
                    T = 0
                    ul = 0
                else:
                    max_val = np.max(block)
                    min_val = np.min(block)
                    result = (min_val.astype(np.uint16) + max_val.astype(np.uint16) + mean) / 3
                    T = result.astype(np.uint8)
                    if(min_val==max_val): 
                        uh = int(min_val)
                        ul = int(min_val)
                    else:
                        if np.any(block > T):
                            uh = int(np.clip(np.mean(block[block > T]), 0, 255))
                        else:
                            uh = 255  
                        if np.any(block < T):
                            ul = int(np.clip(np.mean(block[block < T]), 0, 255))
                        else:
                            ul = 0  
                encoded_data['U_high'].append(to_char(uh))
                encoded_data['U_low'].append(to_char(ul))
                binary_block = (block >= T).astype(np.uint8)
                bit_array_block = bitarray(binary_block.flatten().tolist())
                encoded_data['quantized_data'].append(bit_array_block)
                num+=1
        print("Number of blocks: ",num)
    return encoded_data


def reconstruct_MBTC(encoded_data):
    if encoded_data:
        block_size = encoded_data['block_size']
        img_height, img_width = encoded_data['img_shape']
        uls = encoded_data['U_low']
        uhs = encoded_data['U_high']
        quantized_data = encoded_data['quantized_data']
        reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
        block_id = 0
        for i in range(0, img_height, block_size):
            for j in range(0, img_width, block_size):
                bit_array_block = quantized_data[block_id]
                numpy_array = np.array(bit_array_block.tolist(), dtype=np.uint8)
                binary_block = numpy_array.reshape((block_size, block_size))
                ul = int(uls[block_id])
                uh = int(uhs[block_id])
                binary_block = binary_block.reshape((block_size, block_size))
                reconstructed_block = np.zeros((block_size, block_size), dtype=np.uint8)
                for k in range(block_size):
                    for l in range(block_size):
                        if binary_block[k, l] == 1:
                            reconstructed_block[k, l] = uh
                        else:
                            reconstructed_block[k, l] = ul
                reconstructed_image[i:i + block_size, j:j + block_size] = reconstructed_block
                block_id += 1
    return reconstructed_image


if __name__ =="__main__":
    img = load_image("D:/git/Block_Truncation_Coding/images/synthetic.png")
    if img is not None:
        print("Original Image Shape: ",img.shape)
        mat= np.array([
            [161,160,163,155],
            [161,160,163,155],
            [160,159,154,154],
            [161,158,153,151]
        ], dtype=np.uint8)    
        uh_output_path="D:/git/Block_Truncation_Coding/compressed/uhs.txt"
        ul_output_path="D:/git/Block_Truncation_Coding/compressed/uls.txt"
        blocks_output_path="D:/git/Block_Truncation_Coding/compressed/blocks.txt"
        encoded_data= encode_MBTC(img,block_size=4)
        save_encoded_data(encoded_data,uh_output_path,ul_output_path,blocks_output_path)
        encoded_data=load_encoded_data(uh_output_path,ul_output_path,blocks_output_path,block_size=4)
        encoded_data['img_shape']=img.shape
        reconstructed_image=reconstruct_MBTC(encoded_data)
        save_image(reconstructed_image, "D:/git/Block_Truncation_Coding/images/compressedMBTC_img.png")
        output_path = "D:/git/Block_Truncation_Coding/images/synthetic.png"
        path2="D:/git/Block_Truncation_Coding/images/compressedMBTC_img.png"
        # psnr_value , ps2= calculate_psnr(output_path, path2)
        # print(f"PSNR for original image and compressed image:  {ps2}")