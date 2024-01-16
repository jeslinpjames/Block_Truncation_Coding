import cv2
import numpy as np

def add_speckle_noise(image, intensity):
    if len(image.shape) == 2:
        # For grayscale images
        row, col = image.shape
        gauss = np.random.randn(row, col)
        noisy = image + image * gauss * intensity
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif len(image.shape) == 3:
        # For color images
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        noisy = image + image * gauss * intensity
        return np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported image shape")

def main():
    # Load an image
    img = cv2.imread("D:/git/Block_Truncation_Coding/images/lena.png")

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define a list of intensities for speckle noise
    intensity_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Apply speckle noise with varying intensities and save the results
    for intensity in intensity_values:
        noisy_img = add_speckle_noise(img_gray.copy(), intensity)
        output_path = f'img/noisy_image_intensity_{intensity}.png'
        cv2.imwrite(output_path, noisy_img)
        print(f'Image saved: {output_path}')

if __name__ == "__main__":
    main()
