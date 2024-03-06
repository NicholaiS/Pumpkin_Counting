import os
from rasterio.windows import Window
import cv2
import numpy as np


def get_chunks(data, chunk_size):
    """
    Creates the chunks.
    """
    height, width, channel = data.shape  # Access shape excluding bands
    for y in range(0, height, chunk_size[0]):
        for x in range(0, width, chunk_size[1]):
            # Ensure chunk stays within image boundaries
            end_y = min(y + chunk_size[0], height)
            end_x = min(x + chunk_size[1], width)
            window = Window(x, y, end_x - x, end_y - y)
            yield data[y:end_y, x:end_x], window, x, y
            

def stitch_chunks(chunk_folder, position_file, output_folder, chunk_size):
    """
    Stitches chunks back together, returning a full .png of all chunks.
    """
    # Load position data
    with open(position_file, "r") as file:
        positions = [tuple(map(int, line.strip().split(','))) for line in file]

    # Determine output image size
    height = max(pos[1] + chunk_size[0] for pos in positions)
    width = max(pos[0] + chunk_size[1] for pos in positions)
    channels = 3  # Assuming RGB images

    # Create an empty output image
    output_image = np.zeros((height, width, channels), dtype=np.uint8)

    # Iterate through chunks and stitch them
    for chunk_num, pos in enumerate(positions):
        # Load chunk image
        chunk_path = os.path.join(chunk_folder, f"Chunk_{chunk_num}", f"Chunk_{chunk_num}.jpg")
        chunk_image = cv2.imread(chunk_path)

        # Draw outline around the chunk
        cv2.rectangle(chunk_image, (0, 0), (chunk_image.shape[1] - 1, chunk_image.shape[0] - 1), (0, 0, 255), 1)

        # Add chunk number to the upper left-hand corner
        cv2.putText(chunk_image, str(chunk_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Resize chunk image while preserving aspect ratio
        aspect_ratio = chunk_image.shape[1] / chunk_image.shape[0]

        if aspect_ratio > 1:  # Chunk is wider than tall
            new_width = min(chunk_size[1], chunk_image.shape[1])
            new_height = int(new_width / aspect_ratio)
        else:  # Chunk is taller than wide
            new_height = min(chunk_size[0], chunk_image.shape[0])
            new_width = int(new_height * aspect_ratio)

        chunk_image_resized = cv2.resize(chunk_image, (new_width, new_height))

        # Calculate position in output image
        x, y = pos
        x_end = x + new_width
        y_end = y + new_height

        # Paste resized chunk into output image
        output_image[y:y_end, x:x_end] = chunk_image_resized

    # Save stitched image
    output_path = os.path.join(output_folder, "stitched_image.jpg")
    cv2.imwrite(output_path, output_image)