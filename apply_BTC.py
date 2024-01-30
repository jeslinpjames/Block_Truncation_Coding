import os
from PIL import Image
from BTC_Implementation import encode_BTC,reconstruct_BTC,load_image,save_image,save_encoded_data,load_encoded_data

def apply_btc_to_folder(input_folder, output_folder, block_size=4):
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
            # Encode BTC
            encoded_data = encode_BTC(img, block_size)

            # Save encoded data
            mean_output_path = f"{output_path}_mean.txt"
            variance_output_path = f"{output_path}_variance.txt"
            blocks_output_path = f"{output_path}_blocks.txt"
            save_encoded_data(encoded_data, mean_output_path, variance_output_path, blocks_output_path)

            # Load encoded data
            loaded_data = load_encoded_data(mean_output_path, variance_output_path, blocks_output_path, block_size)
            loaded_data['img_shape'] = img.shape

            # Reconstruct image
            reconstructed_image = reconstruct_BTC(loaded_data)
       
            # Save reconstructed image
            save_image(reconstructed_image, f"{output_path}_reconstructed.bmp")

if __name__ == "__main__":
    input_folder = "D:/git/Block_Truncation_Coding/img"
    output_folder = "D:/git/Block_Truncation_Coding/compressed_images"
    apply_btc_to_folder(input_folder, output_folder, block_size=4)
