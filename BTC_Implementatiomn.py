import cv2
import json
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

def save_encoded_data(encoded_data, path):
    if encoded_data :
        with open(path, 'w')as f:
            json.dump(encoded_data,f)
        print("Encoded data saved successfully")
    else:
        print("Encoded data not found")

def load_encoded_data(path):
    try:
        with open(path,'r')as f:
            encoded_data = json.load(f)
        return encoded_data
    except Exception as e:
        print("Error in loading encoded data")
        print(e)
        return None

def encode_BTC(img, block_size=4):
    if img is not None:
        height, width = img.shape
        count = 0
        blocks = []

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img[i:i+block_size, j:j+block_size]
                blocks.append(block)
                count += 1
            encoded_data={
            'block_size':block_size,
            'thresholds': [],
            'quantized_data': []
        }
        for i, block in enumerate(blocks):
            mean = np.mean(block)
            variance = np.var(block)
            a = int(mean - variance)
            b = int(mean + variance)
            threshold = (a + b) // 2
            binary_block = (block >= threshold).astype(np.uint8)
            encoded_data['thresholds'].append(threshold)
            encoded_data['quantized_data'].append(binary_block.tolist())
        return encoded_data

def reconstruct_image(compressed_blocks, img_height, img_width, block_size):
    reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
    block_idx = 0

    for i in range(0, img_height, block_size):
        for j in range(0, img_width, block_size):
            binary_block = compressed_blocks[block_idx]
            reconstructed_image[i:i+block_size, j:j+block_size] = binary_block
            block_idx += 1

    return reconstructed_image



if __name__=="__main__":
    img = load_image("D:/git/Image_Compression_with_SVD/img_2.jpeg")
    # display_image(img)
    if img is not None:
        zoomed_out_img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        # display_image(zoomed_out_img)
        print("Original Image Shape: ",img.shape)
        matrix = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36]
        ])
        compressed_blocks=encode_BTC(img,4)
        reconstructed_image = reconstruct_image(compressed_blocks, img.shape[0], img.shape[1], 4)
        save_image(reconstructed_image, "D:/git/Image_Compression_with_SVD/compressed_img23.jpeg")

