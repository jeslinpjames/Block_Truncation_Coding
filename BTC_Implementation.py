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
            f.write(msgpack.packb(encoded_data, use_bin_type=True))
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
        encoded_data={
            'block_size':block_size,
            'img_shape': img.shape,
            'mean': [],
            'variance':[],
            'quantized_data': []
        }
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img[i:i+block_size, j:j+block_size]
                mean = np.mean(block)
                variance = np.std(block)
                mean = int(np.clip(mean,0,255).astype(np.uint8))
                variance = int(np.clip(variance,0,255).astype(np.uint8))
                binary_block = (block >= mean).astype(np.uint8)
                # packed_binary_block = np.packbits(binary_block)
                encoded_data['mean'].append(mean)
                encoded_data['variance'].append(variance)
                encoded_data['quantized_data'].append(binary_block.tolist())            
    return encoded_data

def reconstruct_BTC(encoded_data):
    if encoded_data:
        block_size = encoded_data['block_size']
        img_height, img_width = encoded_data['img_shape']
        means = encoded_data['mean']
        variances = encoded_data['variance']
        quantized_data = encoded_data['quantized_data']
        reconstructed_image = np.zeros((img_height, img_width), dtype=np.uint8)
        block_id = 0

        for i in range(0, img_height, block_size):
            for j in range(0, img_width, block_size):
                # packed_binary_block = np.array(quantized_data[block_id], dtype=np.uint8)
                # binary_block = np.unpackbits(packed_binary_block)
                binary_block = np.array(quantized_data[block_id], dtype=np.uint8)
                q = np.sum(binary_block)
                if q != 0 and q != block_size**2:
                    mean = means[block_id]
                    variance = variances[block_id]
                    # q = np.sum(binary_block)
                    m = block_size * block_size

                    # Compute 'a' and 'b'
                    a = int(mean - variance * np.sqrt(q / (m - q)))
                    b = int(mean + variance * np.sqrt(m - q) / q)

                    # Use 'a' and 'b' to reconstruct the block
                    binary_block = binary_block.reshape((block_size, block_size))
                    reconstructed_block = np.zeros((block_size, block_size), dtype=np.uint8)
                    for k in range(block_size):
                        for l in range(block_size):
                            if binary_block[k, l] == 1:  # Changed to single indexing (comma)
                                reconstructed_block[k, l] = b
                            else:
                                reconstructed_block[k, l] = a
                    reconstructed_image[i:i + block_size, j:j + block_size] = reconstructed_block
                    block_id += 1

        return reconstructed_image



if __name__=="__main__":
    img = load_image("D:/git/Block_Truncation_Coding/img_2.jpeg")
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
        save_encoded_data(encoded_data,"D:/git/Block_Truncation_Coding/compressed.json")
        encoded_data=load_encoded_data("D:/git/Block_Truncation_Coding/compressed.json")
        reconstructed_image=reconstruct_BTC(encoded_data)
        save_image(reconstructed_image, "D:/git/Block_Truncation_Coding/compressed_img_2.jpeg")

