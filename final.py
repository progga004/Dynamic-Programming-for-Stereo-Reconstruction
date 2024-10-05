import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#read the two images
image1=cv.imread('conesLeft.ppm')
image2=cv.imread('conesRight.ppm')

#convert it to grayscale
image1=cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
image2=cv.cvtColor(image2,cv.COLOR_BGR2GRAY)

#define im2col and NCC score

def patch_to_vector(patch):
    patch_mean = np.mean(patch)
    patch_std = np.std(patch)
    epsilon = 1e-10 
    normalized_patch = (patch - patch_mean) / (patch_std + epsilon)
    return normalized_patch.flatten()

def extract_patches(row, window_size=5):
    half_window = window_size // 2
    padded_row = np.pad(row, (half_window, half_window), mode='reflect')  
    patches = [padded_row[i:i+window_size] for i in range(len(row) - window_size + 1)]
    return np.array([patch_to_vector(patch) for patch in patches])



def compute_DSI_row(left_row, right_row, window_size=5):
    left_patches = extract_patches(left_row, window_size)
    right_patches = extract_patches(right_row, window_size)
    epsilon=1e-10
    # Normalizing the patches to unit vectors, with epsilon to avoid division by zero
    left_norm = np.linalg.norm(left_patches, axis=1, keepdims=True) + epsilon
    right_norm = np.linalg.norm(right_patches, axis=1, keepdims=True) + epsilon

    left_patches /= left_norm
    right_patches /= right_norm

    # Compute the DSI using matrix multiplication
    dsi_row = 1 - np.dot(left_patches, right_patches.T)
    return dsi_row

def compute_full_DSI(left_image, right_image, window_size=5):
    height, width = left_image.shape
    # Adjust the width for the DSI based on window size
    adjusted_width = width - window_size + 1
    dsi = np.zeros((height, adjusted_width, adjusted_width))  # Adjusted for the window size

    for y in range(height):
        dsi_row = compute_DSI_row(left_image[y], right_image[y], window_size)
        dsi[y, :dsi_row.shape[0], :dsi_row.shape[1]] = dsi_row

    return dsi
def compute_disparity_and_occlusions(path, width, occlusion_value=0):
    disparity_map = np.full(width, occlusion_value, dtype=np.float32) 
    for i, j in path:
        if i < width and j < width:  # Check within bounds
            disparity_map[i] = abs(j - i)  # Compute disparity
    return disparity_map

def fill_occlusions(disparity_map, occlusion_value=0):
    filled_disparity_map = disparity_map.copy()
    for i in range(len(disparity_map)):
        if disparity_map[i] == occlusion_value:
            if i > 0:
                filled_disparity_map[i] = filled_disparity_map[i-1]
    return filled_disparity_map


def visualize_ncc_for_disparity(dsi, disparity):
    ncc_scores = dsi[:, :, disparity]
    normalized_scores = (ncc_scores + 1) / 2

    # Replace NaNs and infinities with 0
    normalized_scores = np.nan_to_num(normalized_scores)

    # Convert to 8-bit image
    image_8bit = (normalized_scores * 255).astype(np.uint8)

    cv.imshow('NCC Scores for Disparity {}'.format(disparity), image_8bit)
    cv.waitKey(0)
    cv.destroyAllWindows()

def visualize_dsi_row_grayscale(dsi, row_number):
    # Extract the DSI for a specific row.
    dsi_row = dsi[row_number, :, :]
    dsi_row_min = np.min(dsi_row)
    dsi_row_max = np.max(dsi_row)
    dsi_row_normalized = (dsi_row - dsi_row_min) / (dsi_row_max - dsi_row_min)
    image_8bit = (dsi_row_normalized * 255).astype(np.uint8)

    cv.imshow(f'DSI for Row {row_number}', image_8bit)
    cv.waitKey(0)
    cv.destroyAllWindows()

def stereo_dynamic_programming(dsi, occlusion_cost):
    height, width = dsi.shape 

    # Initialize the cost matrix C and the move matrix M.
    C = np.zeros((height+1, width+1)) + np.inf  # Add +1 to handle the top and left border.
    M = np.zeros((height+1, width+1), dtype=int)

    # The cost of the top-left corner is 0.
    C[0, 0] = 0

    # The cost of the first row and first column represents occlusion.
    for i in range(1, height+1):
        C[i, 0] = i * occlusion_cost
    for j in range(1, width+1):
        C[0, j] = j * occlusion_cost

    # Compute the cost matrix.
    for i in range(1, height+1):
        for j in range(1, width+1):
            cost_match = C[i-1, j-1] + dsi[i-1, j-1]
            cost_occlusion_i = C[i-1, j] + occlusion_cost
            cost_occlusion_j = C[i, j-1] + occlusion_cost

            # Find the minimum cost.
            cmin = min(cost_match, cost_occlusion_i, cost_occlusion_j)
            C[i, j] = cmin

            # Record the move to the argmin matrix M.
            if cmin == cost_match:
                M[i, j] = 1
            elif cmin == cost_occlusion_i:
                M[i, j] = 2
            elif cmin == cost_occlusion_j:
                M[i, j] = 3

    # Now, we can backtrack from the bottom-right corner to find the path.
    p = []
    i, j = height, width
    while i > 0 and j > 0:
        p.append((i-1, j-1))
        if M[i, j] == 1:
            i -= 1
            j -= 1
        elif M[i, j] == 2:
            i -= 1
        elif M[i, j] == 3:
            j -= 1

    p.reverse()  # Reverse the path to start from the top-left corner.
    return C, M, p



def visualize_path_on_dsi(dsi_row, path):
  
    dsi_image = np.stack((dsi_row,) * 3, axis=-1)
    dsi_norm = (dsi_row - np.min(dsi_row)) / (np.max(dsi_row) - np.min(dsi_row))
    dsi_image = np.uint8(255 * np.stack((dsi_norm, dsi_norm, dsi_norm), axis=-1))
    
   
    height, width, _ = dsi_image.shape
    for i, j in path:
        if i < height and j < width:
            dsi_image[i, j, :] = [255, 0, 0]
        else:
            print(f"Skipping out-of-bounds path coordinates: ({i}, {j})")

    # Display the image with the path overlay.
    plt.imshow(dsi_image)
    plt.title('DSI')
    plt.xlabel('Left Scanline')
    plt.ylabel('Right Scanline')
    plt.show()


def visualize_cost_matrix_grayscale_with_line(C, occlusion_cost, path):
    cost_max = np.max(C[1:, 1:])
    cost_min = np.min(C[1:, 1:])
    normalized_C = (C - cost_min) / (cost_max - cost_min)

    normalized_C[0, :] = 1
    normalized_C[:, 0] = 1

    grayscale_image = (normalized_C * 255).astype(np.uint8)

    image_with_line = grayscale_image.copy()

    for i, j in path:
        image_with_line[i, j] = 255 

    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_line, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    plt.title('Grayscale Cost Matrix with Line')
    plt.xlabel('Right Image Pixels')
    plt.ylabel('Left Image Pixels')
    plt.show()


# Example usage:
def visualize_disparity_map(disparity_map):
    min_disp = np.min(disparity_map[disparity_map > 0])  
    max_disp = np.max(disparity_map)
    normalized_disparity_map = (disparity_map - min_disp) / (max_disp - min_disp)
    normalized_disparity_map *= 255
    normalized_disparity_map[disparity_map == 0] = 0  
    
    # Convert to 8-bit image
    disparity_image = normalized_disparity_map.astype(np.uint8)

    plt.imshow(disparity_image, cmap='gray')
    plt.axis('off') 
    plt.show()

    return disparity_image
def smooth_disparity_map(disparity_map, filter_type='gaussian', kernel_size=5):
    smoothed_disparity_map = cv.GaussianBlur(disparity_map, (kernel_size, kernel_size), 0)
    return smoothed_disparity_map

def compute_and_visualize_full_disparity_map(left_image, right_image, occlusion_cost):
    height, width = left_image.shape
    full_disparity_map = np.zeros((height, width), dtype=np.float32) 
    
    dsi = compute_full_DSI(left_image, right_image)  # Compute the full DSI
    
    for y in range(height):
        dsi_row = dsi[y, :, :]  # Extract the DSI for the current row
        C, M, path = stereo_dynamic_programming(dsi_row, occlusion_cost)  # Stereo DP for current row
        disparity_row = compute_disparity_and_occlusions(path, width)  # Compute disparity for the row
        filled_disparity_row = fill_occlusions(disparity_row)  # Fill occlusions for the row
        full_disparity_map[y, :] = filled_disparity_row  # Store in the full disparity map
    
    # Visualize the full disparity map
    smoothed_disparity_map = smooth_disparity_map(full_disparity_map, filter_type='gaussian', kernel_size=5)

# Visualize the smoothed disparity map
    disparity_image = visualize_disparity_map(smoothed_disparity_map)
   
    
    return disparity_image




occlusion_cost = 0.2 
dsi = compute_full_DSI(image1, image2)  
visualize_ncc_for_disparity(dsi, 10)    
row_number_to_visualize = 50  
visualize_dsi_row_grayscale(dsi, row_number_to_visualize)
dsi_row = dsi[50] 
C, M, path = stereo_dynamic_programming(dsi_row, occlusion_cost)
visualize_path_on_dsi(dsi_row, path)
visualize_cost_matrix_grayscale_with_line(C, occlusion_cost, path)
disparity_image = compute_and_visualize_full_disparity_map(image1, image2, occlusion_cost)


  