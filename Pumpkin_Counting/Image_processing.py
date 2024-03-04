import cv2
import numpy as np

def load_image(file_path):
    """
    Loads image file using OpenCV
    """
    return cv2.imread(file_path)

def calculate_mahalanobis_distance(img, reference_color, covariance_matrix, threshold=5.0):
    """
    Using a reference colour and the associated covariance to find the mahalanobis distance for all pixels.
    The threshold is used to filter out unwanted noice.
    """
    pixels = img.reshape(-1, 3)
    diff = pixels - np.repeat([reference_color], pixels.shape[0], axis=0)
    inv_cov = np.linalg.inv(covariance_matrix)
    mahalanobis_dist = np.sum(diff * (diff @ inv_cov), axis=1)
    mahalanobis_distance_image = mahalanobis_dist.reshape(img.shape[0], img.shape[1])

    # Manual thresholding
    binary_mask = (mahalanobis_distance_image < threshold).astype(np.uint8) * 255
    return binary_mask

def inRange_RGB():
    return None

def inRange_CieLAB():
    return None

def save_image(image, file_path):
    """
    Saves image file using OpenCV
    """
    cv2.imwrite(file_path, image)

def load_txt(file_path):
    """
    Loads a .txt file using Numpy
    """
    return np.loadtxt(file_path)

def extract_features(img_maha, chunk_data, chunk_num, AREA_LOWER=None, AREA_UPPER=None, CIRCULARITY=None):
    """
    Finds and draws contours on the binary image found using the mahalanobis distance.
    For noisy images use the area interval and circularity checks to filter it out.
    """
    # Edge detection using Canny
    tLower = 50
    tUpper = 250
    imgCanny = cv2.Canny(img_maha, tLower, tUpper)
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Drawing the contours around the objects and saving the image
    cv2.drawContours(img_maha, contours, -1, (0, 0, 255), 2)
    save_image(img_maha, f"Chunks/Chunk_{chunk_num}/mahalanobis_dist_image_with_contours.jpg")

    pumpkin_positions_chunk = []  # List to store pumpkin positions in this chunk
    count = 0

    # Counting objects based on area and circularity
    with open(f"Chunks/Chunk_{chunk_num}/Pumpkins_Chunk{chunk_num}_datalog.txt", 'w') as f:
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            # Apply default values if parameters are not provided
            if (CIRCULARITY is None or circularity > CIRCULARITY) and \
               (AREA_LOWER is None or AREA_UPPER is None or (AREA_LOWER < area < AREA_UPPER)):
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    # Finding the centroid for the contours using image moments.
                    cX = int(moments['m10'] / moments['m00'])
                    cY = int(moments['m01'] / moments['m00'])

                    # Logging relevant information for each chunk
                    f.write(f"Contour:\n")
                    f.write(f"Area: {area}\n")
                    f.write(f"Perimeter: {perimeter}\n")
                    f.write(f"Circularity: {circularity}\n")
                    f.write(f"Object center position: x: {cX}, y: {cY}\n")

                    # Add pumpkin position to the list
                    pumpkin_positions_chunk.append((cX, cY))

                    count += 1
        f.write(f"Pumpkins counted: {count}\n")

    # Return chunk data, count, and pumpkin positions
    return chunk_data, count, pumpkin_positions_chunk


