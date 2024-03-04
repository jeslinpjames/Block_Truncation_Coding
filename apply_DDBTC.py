import os
import numpy as np
from DDBTC import encode_DDBTC,reconstruct_DDBTC,load_image,save_image,save_encoded_data,load_encoded_data

def apply_ddbtc_to_folder(input_folder, output_folder, block_size=4):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png','.bmp'))]

    for image_file in image_files:
        # Construct full path for input and output images
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f'compressed_{image_file}')

        # Load the image
        img = load_image(input_path)

        if img is not None:
            class_mat = np.array([[14, 0, 11, 1],
                              [5, 8, 2, 15],
                              [6, 10, 12, 3],
                              [4, 9, 13, 7]])
            diffmat = np.array([[0, 0, 0, 7/16],
                    [3/16, 5/16, 1/16, 0],
                    [0, 0, 0, 0]])
            # Encode DDBTC
            encoded_data = encode_DDBTC(img, block_size, class_matrix=class_mat, diff_matrix=diffmat)

            # Save encoded data
            maxs_output_path = f"{output_path}_maxs.txt"
            mins_output_path = f"{output_path}_mins.txt"
            blocks_output_path = f"{output_path}_blocks.txt"
            save_encoded_data(encoded_data, maxs_output_path, mins_output_path, blocks_output_path)

            # Load encoded data
            loaded_data = load_encoded_data(maxs_output_path, mins_output_path, blocks_output_path, img,block_size)
            loaded_data['img_shape'] = img.shape

            # Reconstruct image
            reconstructed_image = reconstruct_DDBTC(loaded_data)

            # Save reconstructed image
            save_image(reconstructed_image, f"{output_path}_reconstructed.bmp")

if __name__ == "__main__":
    input_folder = "img"
    output_folder = "compressed_images"
    apply_ddbtc_to_folder(input_folder, output_folder, block_size=(4,4))
