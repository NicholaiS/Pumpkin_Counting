import cv2
import numpy as np

def calculate_mahalanobis_distance_manual(img, reference_color, covariance_matrix, threshold=5.0):
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
    segmented_img = (mahalanobis_distance_image < threshold).astype(np.uint8) * 255
    return segmented_img

def calculate_mahalanobis_distance_otsu(img, reference_color, covariance_matrix):
    """
    Using a reference colour and the associated covariance to find the mahalanobis distance for all pixels.
    The threshold is used to filter out unwanted noice.
    """
    pixels = img.reshape(-1, 3)
    diff = pixels - np.repeat([reference_color], pixels.shape[0], axis=0)
    inv_cov = np.linalg.inv(covariance_matrix)
    mahalanobis_dist = np.sum(diff * (diff @ inv_cov), axis=1)
    mahalanobis_distance_image = mahalanobis_dist.reshape(img.shape[0], img.shape[1])

    # Convert the Mahalanobis distance image to 8-bit unsigned integer
    mahalanobis_distance_image_uint8 = cv2.convertScaleAbs(mahalanobis_distance_image)

    # Otsu thresholding
    _, threshold = cv2.threshold(mahalanobis_distance_image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_img = (mahalanobis_distance_image_uint8 < threshold).astype(np.uint8) * 255
    return segmented_img

def color_segmentation_lab(image, lower_bound_lab, upper_bound_lab):
    # Convert image to CIE LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imwrite("Segmentation tests/CieLAB.png", image_lab)

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


chunk_num = 4

loaded_data = np.loadtxt(f"Input files/mask_img_1_covariance_matrix.txt")
reference_color = loaded_data[:3]
covariance_matrix = loaded_data[3:].reshape(3, 3)

image = cv2.imread(f"Chunks/Chunk_{chunk_num}/Chunk_{chunk_num}.png")
print("Data type of the image:", image.dtype)

# Sample BGR from GIMP: 93, 188, 250
lower_bound_bgr = [82, 177, 241]  
upper_bound_bgr = [104, 199, 255]  

# Sample CieLAB from GIMP: 66.3, 23.0, -19
# 8-bit converted: 169.065, 151, 109
lower_bound_lab = [139, 145, 104]  
upper_bound_lab = [189, 191, 169]

# Perform color segmentation using: inRange with BGR values
mask_lab = color_segmentation_lab(image, lower_bound_lab, upper_bound_lab)

# Perform color segmentation using: inRange with CieLAB values
mask_bgr = color_segmentation_bgr(image, lower_bound_bgr, upper_bound_bgr)

# Perform color segmentation using: Distance in BGR space to a reference colour
mask_maha = calculate_mahalanobis_distance_manual(image, reference_color, covariance_matrix, 10.0)

mask_maha_otsu = calculate_mahalanobis_distance_otsu(image, reference_color, covariance_matrix)

# Display the segmented masks
cv2.imwrite('Segmentation tests/Segmented Mask (BGR).png', mask_bgr)
cv2.imwrite('Segmentation tests/Segmented Mask (CIE LAB).png', mask_lab)
cv2.imwrite('Segmentation tests/Segmented Mask (Mahalanobis Distance Manual threshold).png', mask_maha)
cv2.imwrite('Segmentation tests/Segmented Mask (Mahalanobis Distance Otsu threshold).png', mask_maha_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
