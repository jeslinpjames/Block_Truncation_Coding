import cv2

def convert_and_save_grayscale(input_path, output_path):
    try:
        # Load the image in color
        color_image = cv2.imread(input_path)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        cv2.imwrite(output_path, grayscale_image)

        print("Grayscale image saved successfully at:", output_path)

    except Exception as e:
        print("Error converting and saving the image:")
        print(e)

# Example usage:
input_path="D:/git/Block_Truncation_Coding/images/lena.png"
output_path = "D:/git/Block_Truncation_Coding/images/lena2.png"

convert_and_save_grayscale(input_path, output_path)
