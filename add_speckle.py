import cv2
import numpy as np

def add_speckle_noise(image, intensity):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    noisy = image + image * gauss * intensity
    return np.clip(noisy, 0, 255).astype(np.uint8)

def main():
    # Load an image
    img = cv2.imread("D:/git/Block_Truncation_Coding/images/synthetic.png")

    # Define a list of intensities for speckle noise
    intensity_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Apply speckle noise with varying intensities and save the results
    for intensity in intensity_values:
        noisy_img = add_speckle_noise(img.copy(), intensity)
        output_path = f'img/noisy_image_intensity_{intensity}.png'
        cv2.imwrite(output_path, noisy_img)
        print(f'Image saved: {output_path}')

if __name__ == "__main__":
    main()
