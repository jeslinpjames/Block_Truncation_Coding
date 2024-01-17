import cv2
import os
import numpy as np
import tensorflow as tf


def calculate_psnr(original_image_path, compressed_image_path):
    # Read the images
    original_image = cv2.imread(original_image_path)
    compressed_image = cv2.imread(compressed_image_path)

    # Convert images to float32 for PSNR calculation
    original_image = original_image.astype(float)
    compressed_image = compressed_image.astype(float)

    # Calculate the MSE (Mean Squared Error)
    mse = ((original_image - compressed_image) ** 2).mean()

    # Calculate the PSNR
    if mse == 0:
        return float('inf')
    psnr = 20 * (max(original_image.flatten()) / mse) ** 0.5
    psnr2 = tf.image.psnr(original_image, compressed_image, max_val=255)
    return psnr,psnr2

# Specify the folder containing the images
folder_path = "D:/git/Block_Truncation_Coding/img"
if __name__ == "__main__":
    # Iterate over intensity values (0.1 to 0.9)
    for intensity in range(1, 10):
        intensity_value = intensity / 10.0
        original_image_path = os.path.join(folder_path, f"noisy_image_intensity_{intensity_value:.1f}.png")
        compressed_image_path = os.path.join(folder_path, f"compressed_noisy_image_intensity_{intensity_value:.1f}.png_reconstructed.png")

        # Calculate PSNR
        psnr_value , ps2= calculate_psnr(original_image_path, compressed_image_path)

        # Print the result
        # print(f"PSNR for intensity {intensity_value:.1f}: {psnr_value}")
        print(f"PSNR for intensity  {intensity_value:.1f}: {ps2}")

    img1 = "D:\git\Block_Truncation_Coding\images\synthetic.png"
    img2 = "D:\git\Block_Truncation_Coding\images\compressedMBTC_img.png"
    psnr_value , ps2= calculate_psnr(img1, img2)
    print(f"PSNR for original image and compressed image:  {intensity_value:.1f}: {ps2}")
