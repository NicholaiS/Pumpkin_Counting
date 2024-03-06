# ---------------------------------------------------------- Imports -------------------------------------------------------------
import os
import rasterio
import cv2
from Chunk_manipulation import stitch_chunks, get_chunks
from Image_processing import save_image, load_image, load_txt, calculate_mahalanobis_distance, extract_features

# ------------------------------------------------------ Dirs & Constants ---------------------------------------------------------
input_folder = "Chunks"
position_file = "Position data/chunk_positions.txt"
output_folder = "Outputs"

MASK_NO = 3
THRESHOLD = 10.0
DUBLICATE_THRESHOLD = 4

# --------------------------------------------------------- Main Loop -------------------------------------------------------------
def main():
    pumpkin_count = 0
    pumpkin_positions = []
    
    loaded_data = load_txt(f"Input files/mask_img_{MASK_NO}_covariance_matrix.txt")
    reference_color = loaded_data[:3]
    covariance_matrix = loaded_data[3:].reshape(3, 3)
    
    with rasterio.open('Input files/PumpkinField.tif') as src:
        # Get image data and window information
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)

        bgr = cv2.merge([blue, green, red])

        # Define chunk size
        chunk_size = (1000, 1000)

        # Create output folder if not exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get chunks and save them
        with open(position_file, "w") as log_file:
            for chunk_num, (chunk_data, window, x, y) in enumerate(get_chunks(bgr, chunk_size)):
                chunk_pos = (x, y)
                # Save chunk in separate folder
                chunk_folder = os.path.join(input_folder, f"Chunk_{chunk_num}")
                os.makedirs(chunk_folder, exist_ok=True)
                chunk_path = os.path.join(chunk_folder, f"Chunk_{chunk_num}.jpg")
                cv2.imwrite(chunk_path, chunk_data)
            
                # Log chunk position
                log_file.write(f"{x},{y}\n")
                
                # Color segmentation
                mahalanobis_distance_image = calculate_mahalanobis_distance(chunk_data, reference_color, covariance_matrix, THRESHOLD)
                save_image(mahalanobis_distance_image, f"Chunks/Chunk_{chunk_num}/mahalanobis_dist_image.jpg")
                
                # Feature extraction
                img_maha = load_image(f"Chunks/Chunk_{chunk_num}/mahalanobis_dist_image.jpg")
                chunk_data, pumpkin_positions_chunk = extract_features(img_maha, chunk_data, chunk_num)
                
                # Adjust pumpkin positions with chunk coordinates
                adjusted_pumpkin_positions = [(x + chunk_pos[0], y + chunk_pos[1]) for (x, y) in pumpkin_positions_chunk]
                
                # Iterate through each adjusted position from the chunk
                for adjusted_pos in adjusted_pumpkin_positions:
                    is_duplicate = False
    
                    # Iterate through each existing pumpkin position
                    for pos in pumpkin_positions:
                        # Check if the adjusted position is within +-5 in both x and y directions of any existing position
                        if pos[0] - DUBLICATE_THRESHOLD <= adjusted_pos[0] <= pos[0] + DUBLICATE_THRESHOLD \
                        and pos[1] - DUBLICATE_THRESHOLD <= adjusted_pos[1] <= pos[1] + DUBLICATE_THRESHOLD:
                            is_duplicate = True
                            break  # No need to continue checking if a duplicate is found
    
                    # If the adjusted position is not a duplicate, add it to the pumpkin_positions dictionary
                    if not is_duplicate:
                        pumpkin_positions.append(adjusted_pos)
                        
    stitch_chunks(input_folder, position_file, output_folder, chunk_size)
    field = load_image("Outputs/stitched_image.jpg")
    
    # Iterate through the pumpkin_positions dictionary
    for pos in pumpkin_positions:
        pumpkin_count = pumpkin_count + 1
        cv2.circle(field, pos, 5, (0, 0, 255), 1)
        
    save_image(field, "Outputs/stitched_image_marked.jpg")

    print("Pumpkins estimated to be in the field: ", pumpkin_count)

if __name__ == "__main__":
    main()