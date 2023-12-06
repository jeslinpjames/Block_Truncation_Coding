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

def load_encoded_data(mean_path, variance_path, blocks_path, block_size=4):
    try:
        encoded_data = {
            'block_size': block_size,
            'img_shape': None,
            'means': [],
            'variances': [],
            'quantized_data': []
        }
        with open(mean_path, 'rb') as f:
            encoded_data['means'] = list(f.read())
        with open(variance_path, 'rb') as f:
            encoded_data['variances'] = list(f.read())
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

def reconstruct_BTC(encoded_data):
    if encoded_data:
        block_size = encoded_data['block_size']
        img_height, img_width = encoded_data['img_shape']
        means = encoded_data['means']
        variances = encoded_data['variances']
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
                variance = variances[block_id]
                m = block_size * block_size
                q = np.sum(binary_block)
                m = block_size * block_size
                if q != 0 and q != block_size**2:
                    # Compute 'a' and 'b'
                    a = int(mean - variance * np.sqrt(q / (m - q)))
                    b = int(mean + variance * np.sqrt(m - q) / q)
                else:
                    # Compute 'a' and 'b'
                    a = int(mean - variance)
                    b = int(mean + variance)
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


if __name__=="__main__":
    img = load_image("D:/git/Block_Truncation_Coding/images/synthetic.png")
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
        mean_output_path="D:/git/Block_Truncation_Coding/compressed/mean.txt"
        variance_output_path="D:/git/Block_Truncation_Coding/compressed/variance.txt"
        blocks_output_path="D:/git/Block_Truncation_Coding/compressed/blocks.txt"
        encoded_data= encode_BTC(img,block_size=4)
        save_encoded_data(encoded_data,mean_output_path,variance_output_path,blocks_output_path)
        encoded_data=load_encoded_data(mean_output_path,variance_output_path,blocks_output_path,block_size=4)
        encoded_data['img_shape']=img.shape
        reconstructed_image=reconstruct_BTC(encoded_data)
        save_image(reconstructed_image, "D:/git/Block_Truncation_Coding/images/compressed_img.jpeg")

    
