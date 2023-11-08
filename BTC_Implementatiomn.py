import cv2
import msgpack
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
        with open(path, 'wb')as f:
            f.write(msgpack.packb(encoded_data, use_bin_type=True)))
        print("Encoded data saved successfully")
    else:
        print("Encoded data not found")

def load_encoded_data(path):
    try:
        with open(path,'rb')as f:
            encoded_data = msgpack.unpackb(f.read(), raw=False)
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

# def reconstruct_BTC(encoded_data):
#     if encoded_data:
#         block_size = encoded_data['block_size']
#         thresholds = encoded_data['thresholds']
#         quantized_data = encoded_data['quantized_data']
#         img_height = block_size * len(quantized_data)
#         img_width = block_size * len(quantized_data[0])

#         reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
#         block_id = 0

#         for i in range(0,img_height,block_size):
#             for j in range(0, img_width,block_size):
#                 threshold = thresholds[block_id]
#                 binary_block = np.array(quantized_data[block_id])
#                 a = threshold - (block_size // 2)
#                 b = threshold + (block_size // 2)
#                 reconstructed_block = (binary_block * (b - a) + a).astype(np.uint8)
#                 reconstructed_image[i:i+block_size, j:j+block_size] = reconstructed_block
#             block_id += 1
#     return reconstructed_image

def reconstruct_BTC(encoded_data):
    if encoded_data:
        block_size = encoded_data['block_size']
        thresholds = encoded_data['thresholds']
        quantized_data = encoded_data['quantized_data']
        img_height = block_size * len(quantized_data)
        img_width = block_size * len(quantized_data[0])

        reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
        block_id = 0

        for i in range(0, img_height, block_size):
            for j in range(0, img_width, block_size):
                threshold = thresholds[block_id]
                binary_block = np.array(quantized_data[block_id])
                a = threshold - (block_size // 2)
                b = threshold + (block_size // 2)
                reconstructed_block = (binary_block * (b - a) + a).astype(np.uint8)
                reconstructed_image[i:i+block_size, j:j+block_size] = reconstructed_block
                block_id += 1  
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
        encoded_data=encode_BTC(img,4)
        # save_encoded_data(encoded_data,"D:/git/Image_Compression_with_SVD/encoded_data.json")
        encoded_data=load_encoded_data("D:/git/Image_Compression_with_SVD/encoded_data.json")
        reconstructed_image=reconstruct_BTC(encoded_data)
        save_image(reconstructed_image, "D:/git/Image_Compression_with_SVD/compressed_img23.jpeg")

