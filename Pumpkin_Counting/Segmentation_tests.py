import cv2
import numpy as np

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

def color_segmentation_lab(image, lower_bound_lab, upper_bound_lab):
    # Convert image to CIE LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lower_bound_lab = np.array(lower_bound_lab)
    upper_bound_lab = np.array(upper_bound_lab)

    # Create a mask using CIE LAB color space
    mask_lab = cv2.inRange(image_lab, lower_bound_lab, upper_bound_lab)

    return mask_lab

def color_segmentation_bgr(image, lower_bound_bgr, upper_bound_bgr):
    lower_bound_bgr = np.array(lower_bound_bgr)
    upper_bound_bgr = np.array(upper_bound_bgr)

    # Create a mask using BGR color space
    mask_bgr = cv2.inRange(image, lower_bound_bgr, upper_bound_bgr)

    return mask_bgr


chunk_num = 2

loaded_data = np.loadtxt(f"Input files/mask_img_1_covariance_matrix.txt")
reference_color = loaded_data[:3]
covariance_matrix = loaded_data[3:].reshape(3, 3)

image = cv2.imread(f"Chunks/Chunk_{chunk_num}/Chunk_{chunk_num}.png")

# Sample BGR from GIMP: 93, 188, 250
lower_bound_bgr = [82, 177, 241]  
upper_bound_bgr = [104, 199, 255]  

# Convert BGR to LAB using OpenCV
lower_bound_lab_opencv = cv2.cvtColor(np.uint8([[lower_bound_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
upper_bound_lab_opencv = cv2.cvtColor(np.uint8([[upper_bound_bgr]]), cv2.COLOR_BGR2LAB)[0][0]

# Perform color segmentation using: inRange with RGB values
mask_lab = color_segmentation_lab(image, lower_bound_lab_opencv, upper_bound_lab_opencv)

# Perform color segmentation using: inRange with CieLAB values
mask_bgr = color_segmentation_bgr(image, lower_bound_bgr, upper_bound_bgr)

# Perform color segmentation using: Distance in RGB space to a reference colour
mask_maha = calculate_mahalanobis_distance(image, reference_color, covariance_matrix, 10.0)

# Display the segmented masks
cv2.imshow('Original', image)
cv2.imshow('Segmented Mask (BGR)', mask_bgr)
cv2.imshow('Segmented Mask (CIE LAB)', mask_lab)
cv2.imshow('Segmented Mask (Mahalanobis Distance and ref color)', mask_maha)
cv2.waitKey(0)
cv2.destroyAllWindows()
