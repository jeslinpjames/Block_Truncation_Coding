import cv2
import numpy as np

def compare_images(img1, img2):
    if img1.shape == img2.shape:
        diff_locations = np.where(img1 != img2)

        if len(diff_locations[0]) > 0:
            print("Differing pixels:")
            for i in range(len(diff_locations[0])):
                row, col = diff_locations[0][i], diff_locations[1][i]
                pixel_value_img1 = img1[row, col]
                pixel_value_img2 = img2[row, col]
                print(f"At pixel ({row}, {col}): Image 1 value = {pixel_value_img1}, Image 2 value = {pixel_value_img2}")

        else:
            print("Both images are identical.")
    else:
        print("Images have different shapes.")

if __name__ == "__main__":
    # Load the original and reconstructed images
    original_img = cv2.imread("D:/git/Block_Truncation_Coding/images/synthetic.bmp", cv2.IMREAD_GRAYSCALE)
    compressed_img = cv2.imread("D:/git/Block_Truncation_Coding/images/compressedAMBTC_img.bmp", cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded successfully
    if original_img is not None and compressed_img is not None:
        # Compare the images
        compare_images(original_img, compressed_img)
    else:
        print("Error loading images.")
