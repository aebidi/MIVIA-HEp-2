import os
import cv2

# Function to load full images and corresponding full masks
def load_full_data(image_dir):
    images = []
    masks = []

    for folder in sorted(os.listdir(image_dir)):
        if os.path.isdir(os.path.join(image_dir, folder)):
            image_path = os.path.join(image_dir, folder, folder + '.bmp')
            mask_path = os.path.join(image_dir, folder, folder + '_mask.bmp')
            if os.path.exists(image_path) and os.path.exists(mask_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                masks.append(mask)

    return images, masks

# Function to load individual cell images and corresponding cell masks
def load_cell_data(cell_images_dir, cell_masks_dir):
    cell_images = []
    cell_masks = []

    for folder in sorted(os.listdir(cell_images_dir)):
        if os.path.isdir(os.path.join(cell_images_dir, folder)):
            cell_images_per_image = []
            cell_masks_per_image = []
            for filename in sorted(os.listdir(os.path.join(cell_images_dir, folder))):
                cell_image_path = os.path.join(cell_images_dir, folder, filename)
                cell_mask_path = os.path.join(cell_masks_dir, folder, filename)
                if os.path.exists(cell_image_path) and os.path.exists(cell_mask_path):
                    cell_image = cv2.imread(cell_image_path, cv2.IMREAD_GRAYSCALE)
                    cell_mask = cv2.imread(cell_mask_path, cv2.IMREAD_GRAYSCALE)
                    cell_images_per_image.append(cell_image)
                    cell_masks_per_image.append(cell_mask)
            cell_images.append(cell_images_per_image)
            cell_masks.append(cell_masks_per_image)

    return cell_images, cell_masks

# Function to draw boundaries around cells based on cell masks
def draw_boundaries(images, masks, cell_images, cell_masks):
    bordered_images = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    # Iterate through each image and its corresponding full mask
    for image, mask, cell_images_per_image, cell_masks_per_image in zip(images, masks, cell_images, cell_masks):
        # Draw boundaries around cells
        bordered_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing color boundaries
        for cell_mask in cell_masks_per_image:
            # Find contours
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours on the image
            cv2.drawContours(bordered_image, contours, -1, (0, 255, 0), 1)
        
        # Compare with ground truth full mask
        intersection = cv2.countNonZero(cv2.bitwise_and(cell_masks_per_image[0], mask))
        union = cv2.countNonZero(cv2.bitwise_or(cell_masks_per_image[0], mask))
        iou = intersection / union
        
        if iou > 0.5:
            true_positives += 1
        else:
            false_positives += 1
    
    for cell_masks_per_image in cell_masks:
        intersection = cv2.countNonZero(cv2.bitwise_and(cell_masks_per_image[0], mask))
        union = cv2.countNonZero(cv2.bitwise_or(cell_masks_per_image[0], mask))
        iou = intersection / union
        
        if iou <= 0.5:
            false_negatives += 1

    # True negatives calculation (assuming we have full image masks in the 'masks' variable)
    union = cv2.countNonZero(cv2.bitwise_or(mask, 255 - mask))
    true_negatives += (image.size - union) / 255 - true_positives
    
    bordered_images.append(bordered_image)
    
    return bordered_images, true_positives, false_positives, false_negatives, true_negatives

# Load full images and masks
full_image_dir = 'Images'  # Directory containing full images
images, masks = load_full_data(full_image_dir)

# Load individual cell images and masks
cell_images_dir = 'Cells/Cell_Images'   # Directory containing individual cell images
cell_masks_dir = 'Cells/Cell_Masks'     # Directory containing individual cell masks
cell_images, cell_masks = load_cell_data(cell_images_dir, cell_masks_dir)

# Draw boundaries around cells based on cell masks and calculate metrics
bordered_images, true_positives, false_positives, false_negatives, true_negatives = draw_boundaries(images, masks, cell_images, cell_masks)

for image, mask, cell_images_per_image, cell_masks_per_image in zip(images, masks, cell_images, cell_masks):
    # Draw boundaries around cells
    bordered_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing color boundaries
    for cell_mask in cell_masks_per_image:
        # Find contours
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the image
        cv2.drawContours(bordered_image, contours, -1, (0, 255, 0), 1)
    
    # Display the image with cell boundaries
    smaller_image = cv2.resize(bordered_image, (800, 600))
    cv2.imshow('Image with Cell Boundaries', smaller_image)
    cv2.waitKey(0)
# Calculate accuracy
total_cells = true_positives + false_positives + false_negatives + true_negatives
accuracy = (true_positives + true_negatives) / total_cells if total_cells != 0 else 0

# Calculate precision, recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print or display results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

