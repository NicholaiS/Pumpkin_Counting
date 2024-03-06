import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

REF_IMG_NO = 4

def extract_annotated_pixels_and_mask(original_image_path, annotated_image_path):
    # Read the original and annotated images
    original_image = cv2.imread(original_image_path)
    annotated_image = cv2.imread(annotated_image_path)

    # Create a mask for the annotated pixels (using red color range)
    mask = cv2.inRange(annotated_image, (0, 0, 245), (10, 10, 255))

    # Apply the mask to the original image to extract annotated pixel values
    annotated_pixel_values = original_image[mask == 255]

    return annotated_pixel_values, mask

def calculate_color_statistics(pixel_values):
    # Convert pixel values to numpy array
    pixel_values = np.array(pixel_values)

    # Calculate mean and standard deviation in RGB color space
    mean_rgb = np.mean(pixel_values, axis=0)
    std_rgb = np.std(pixel_values, axis=0)

    # Convert pixel values to CieLAB color space
    pixel_lab = cv2.cvtColor(pixel_values.reshape(1, -1, 3), cv2.COLOR_RGB2Lab)

    # Calculate mean and standard deviation in CieLAB color space
    mean_lab = np.mean(pixel_lab, axis=(0, 1))
    std_lab = np.std(pixel_lab, axis=(0, 1))
    
    # Calculate covariance matrix in RGB color space
    cov_rgb = np.cov(pixel_values.transpose())
    
    full_cov = np.concatenate((mean_rgb, cov_rgb.flatten()))

    return mean_rgb, std_rgb, mean_lab, std_lab, full_cov

def visualize_distribution(pixel_values, pixel_lab, save_path=None):
    # Visualize distribution of color values in RGB and CieLAB color spaces
    r, g, b = pixel_values[:, 0], pixel_values[:, 1], pixel_values[:, 2]
    l, a, b_lab = pixel_lab[:, :, 0], pixel_lab[:, :, 1], pixel_lab[:, :, 2]

    plt.figure(figsize=(12, 6))

    # Plot RGB distribution
    plt.subplot(2, 3, 1)
    plt.hist(r, bins=20, color='red', alpha=0.5, label='R')
    plt.title("RGB Distribution (R)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.hist(g, bins=20, color='green', alpha=0.5, label='G')
    plt.title("RGB Distribution (G)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.hist(b, bins=20, color='blue', alpha=0.5, label='B')
    plt.title("RGB Distribution (B)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    # Plot CieLAB distribution
    plt.subplot(2, 3, 4)
    plt.hist(l.ravel(), bins=20, color='lightgray', alpha=0.5, label='L*')
    plt.title("CieLAB L* Distribution")
    plt.xlabel("L* Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 5)
    plt.hist(a.ravel(), bins=20, color='lightgreen', alpha=0.5, label='a*')
    plt.title("CieLAB a* Distribution")
    plt.xlabel("a* Value")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 6)
    plt.hist(b_lab.ravel(), bins=20, color='skyblue', alpha=0.5, label='b*')
    plt.title("CieLAB b* Distribution")
    plt.xlabel("b* Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def save_mean_color(mean_rgb, save_path):
    mean_rgb_img = np.full((100, 100, 3), mean_rgb, dtype=np.uint8)
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(mean_rgb_img, cv2.COLOR_BGR2RGB))
    plt.title("Mean Color (RGB)")
    plt.axis('off')
    plt.savefig(save_path)

def save_covariance_matrix(cov_rgb, save_path):
    np.savetxt(save_path, cov_rgb)

# Main function
if __name__ == "__main__":
    # Provide the paths to the original and annotated images
    original_image_path = "Mask data/mask_img.jpg"
    annotated_image_path = "Mask data/mask_img_" + str(REF_IMG_NO) + "_red.jpg"

    # Extract annotated pixels and mask
    annotated_pixel_values, mask = extract_annotated_pixels_and_mask(original_image_path, annotated_image_path)

    # Calculate color statistics
    mean_rgb, std_rgb, mean_lab, std_lab, cov_rgb = calculate_color_statistics(annotated_pixel_values)
    print("RGB Mean:", mean_rgb)
    print("RGB Standard Deviation:", std_rgb)
    print("CieLAB Mean:", mean_lab)
    print("CieLAB Standard Deviation:", std_lab)
    print("RGB Covariance Matrix:", cov_rgb)

    # Save the distribution plots
    visualize_distribution(annotated_pixel_values, cv2.cvtColor(annotated_pixel_values.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab), save_path="Mask data/mask_img_" + str(REF_IMG_NO) + "_distribution_plots.jpg")

    # Save the mean color in RGB
    save_mean_color(mean_rgb, save_path="Mask data/mask_img_" + str(REF_IMG_NO) + "_mean_color_rgb.jpg")
    
    if not os.path.exists('Input files'):
        os.makedirs('Input files')
    # Save the covariance matrix
    save_covariance_matrix(cov_rgb, save_path="Input files/mask_img_" + str(REF_IMG_NO) + "_covariance_matrix.txt")
