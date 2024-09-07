from PIL import Image
import os

def rotate_image(file_name):
    # Open the image
    image = Image.open(file_name)

    # Rotate the image by 90 degrees
    rotated_image = image.transpose(Image.ROTATE_90)

    # Add the prefix "90d" to the file name
    rotated_file_name = "90d" + file_name

    # Save the rotated image with the new file name
    rotated_image.save(rotated_file_name)

# Specify the path to your image file
image_path = "Predictions.png"

# Call the rotate_image function
rotate_image(image_path)