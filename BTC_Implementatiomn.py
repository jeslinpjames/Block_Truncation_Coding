import cv2
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

def BTC(img, block_size=4):
    if img is not None:
        height, width = img.shape
        count = 0
        blocks = []

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = img[i:i+block_size, j:j+block_size]
                print("_________ ", count, " ___________")
                print(block)
                blocks.append(block)
                count += 1
       
        


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
        BTC(img,10)
