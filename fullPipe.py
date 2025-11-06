import cv2
import numpy as np
import os
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate

# ---------------------------- #
# Step 1: Skew Correction
# ---------------------------- #
def correct_skew(input_image_path, output_path):
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Could not load {input_image_path} for skew correction.")
    angle = get_angle(image)
    corrected = rotate(image, angle)
    cv2.imwrite(output_path, corrected)
    print(f"✅ Skew correction done -> {output_path}")
    return output_path


# ---------------------------- #
# Step 2: SIFT Alignment
# ---------------------------- #
def align_documents_sift(template_path, filled_doc_path, output_path):
    template_color = cv2.imread(template_path)
    filled_doc_color = cv2.imread(filled_doc_path)
    if template_color is None or filled_doc_color is None:
        raise ValueError("❌ Could not load images. Check file paths.")
    
    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
    filled_doc_gray = cv2.cvtColor(filled_doc_color, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(template_gray, None)
    kp2, desc2 = sift.detectAndCompute(filled_doc_gray, None)
    if desc1 is None or desc2 is None:
        raise ValueError("❌ Could not find enough features in images.")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good_matches) < 10:
        raise ValueError("❌ Not enough good matches found for alignment.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if homography is None:
        raise ValueError("❌ Could not compute homography.")

    h, w, _ = template_color.shape
    aligned_image = cv2.warpPerspective(filled_doc_color, homography, (w, h))
    cv2.imwrite(output_path, aligned_image)
    print(f"✅ SIFT alignment done -> {output_path}")
    return output_path


# ---------------------------- #
# Step 3: Perspective Correction
# ---------------------------- #
def correct_perspective(input_path, output_path="3_corrTab.png"):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"❌ Could not load {input_path} for perspective correction.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corrected = image

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            src_points = np.float32([p[0] for p in approx])
            s = src_points.sum(axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = src_points[np.argmin(s)]
            rect[2] = src_points[np.argmax(s)]
            diff = np.diff(src_points, axis=1)
            rect[1] = src_points[np.argmin(diff)]
            rect[3] = src_points[np.argmax(diff)]

            width, height = 1545, 2000
            dst_points = np.float32([[0, 0], [width-1, 0],
                                     [width-1, height-1], [0, height-1]])
            M = cv2.getPerspectiveTransform(rect, dst_points)
            corrected = cv2.warpPerspective(image, M, (width, height))
    cv2.imwrite(output_path, corrected)
    print(f"✅ Perspective correction done -> {output_path}")
    return output_path


# ---------------------------- #
# Step 4: Remove Red Lines
# ---------------------------- #
def remove_colored_lines(input_image_path="3_corrTab.png", output_path="done.png",
                         hsv_red_min_sat=20, lab_ab_thresh=4,
                         dilate_iters=3, inpaint_radius=3):
    img_bgr = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError("❌ Could not load image for red line removal!")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, hsv_red_min_sat, 30], dtype=np.uint8)
    upper_red1 = np.array([25, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([155, hsv_red_min_sat, 30], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_hsv_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    a, b = lab[:, :, 1] - 128, lab[:, :, 2] - 128
    mask_lab = ((np.abs(a) > lab_ab_thresh) | (np.abs(b) > lab_ab_thresh)).astype(np.uint8) * 255

    combined_mask = cv2.bitwise_or(mask_hsv_red, mask_lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(combined_mask, kernel, iterations=dilate_iters)
    inpainted = cv2.inpaint(img_bgr, mask_dilated, inpaint_radius, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray)
    print(f"✅ Red lines removed -> {output_path}")
    return output_path


# ---------------------------- #
# Step 5: Enhance Handwriting Final
# ---------------------------- #
# ---------------------------- #
# Step 5: Enhance Handwriting Final (Updated)
# ---------------------------- #
def enhance_handwriting_final(image_path: str, output_path: str, morph_kernel_size: int = 3):
    """
    Applies aggressive enhancement to the cleaned image to achieve
    crisp, white-on-black strokes (updated version).
    """
    print("\n--- Step 5: Enhancing Handwriting to White-on-Black Strokes ---")
    try:
        # Load and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"❌ Could not load image at {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_enh = clahe.apply(gray)

        # Gentle Gaussian blur for denoising
        gray_blur = cv2.GaussianBlur(gray_enh, (3, 3), 0)

        # Adaptive threshold (white text on black)
        adaptive_binary = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            61,
            15
        )

        # Morphological cleanup and stroke enhancement
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        opened = cv2.morphologyEx(adaptive_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        final_enhanced = cv2.GaussianBlur(dilated, (3, 3), 0)

        # Save result
        cv2.imwrite(output_path, final_enhanced)
        print(f"✅ Handwriting enhancement complete -> {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ Error during handwriting enhancement: {e}")
        raise



# ---------------------------- #
# Step 6: Extract Valid Letter Boxes
# ---------------------------- #
def extract_valid_letter_boxes(input_image_path, output_dir="valid_letter_boxes"):
    print("\n--- Step 6: Extract Valid Letter Boxes ---")

    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Could not load {input_image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 100 and h >= 100:
            boxes.append((x, y, w, h))

    if not boxes:
        raise ValueError("❌ No boxes detected — adjust filters.")

    widths, heights = np.array([b[2] for b in boxes]), np.array([b[3] for b in boxes])
    w_min, w_max = np.median(widths) * 0.8, np.median(widths) * 1.2
    h_min, h_max = np.median(heights) * 0.8, np.median(heights) * 1.2

    filtered = [b for b in boxes if w_min <= b[2] <= w_max and h_min <= b[3] <= h_max]
    filtered = sorted(filtered, key=lambda b: (b[1] // 50, b[0]))
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    for i, (x, y, w, h) in enumerate(filtered, 1):
        cropped = gray[y:y+h, x:x+w]
        _, bw = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY_INV)
        if cv2.countNonZero(bw) / (w * h) > 0.02:
            cv2.imwrite(os.path.join(output_dir, f"box_{saved+1:02d}.png"), cropped)
            saved += 1
    print(f"✅ Saved {saved} valid boxes in '{output_dir}/'.")

import cv2
import numpy as np
import os

import cv2
import numpy as np
import os

def isolate_character_and_measure(image_path, output_path="isolated_character_output.png"):
    """
    Isolates the character in the image (assuming it's a hole or feature within a larger frame), 
    measures its dimensions, and saves the cropped result.
    """
    # --- Configuration ---
    INPUT_FILENAME = image_path  # use the actual image passed to the function
    
    # 1. Load the image in grayscale
    img_gray = cv2.imread(INPUT_FILENAME, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print(f"Error: Could not load image at '{INPUT_FILENAME}'.")
        print("Please ensure the file is available in the same directory as the script.")
        return

    # 2. Invert and apply thresholding
    _, binary_inverted = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours
    contours, _ = cv2.findContours(binary_inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter contours to exclude the bounding box (largest contour) and small noise
    if not contours:
        print("No significant contours found. Check the threshold value (150).")
        return

    contour_areas = [(cv2.contourArea(c), c) for c in contours]
    min_area = 100 
    valid_contours = [item for item in contour_areas if item[0] > min_area]

    if not valid_contours:
        print("No character contours found after filtering.")
        return
        
    valid_contours.sort(key=lambda x: x[0], reverse=True)

    # ✅ FIXED: Combine all contours except the largest (outer box)
    if len(valid_contours) > 1:
        char_contours = [item[1] for item in valid_contours[1:]]  # skip biggest (frame)
    else:
        char_contours = [valid_contours[0][1]]

    # Merge all points from contours to get one bounding box for the whole character
    all_points = np.vstack(char_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # 5. Optional padding
    padding = 5
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img_gray.shape[1], x + w + padding)
    y_end = min(img_gray.shape[0], y + h + padding)

    # 6. Crop
    isolated_char = img_gray[y_start:y_end, x_start:x_end]

    # 7. Print measured dimensions
    print("--- Character Measurement Results ---")
    print(f"Input File: {INPUT_FILENAME}")
    print(f"Output File: {output_path}")
    print("-" * 35)
    print(f"Isolated Character Dimensions (Width x Height): {w} x {h} pixels")

    # 8. Save isolated image
    cv2.imwrite(output_path, isolated_char)
    print(f"\nIsolated image saved successfully to: {output_path}")


# ---------------------------- #
# MAIN PIPELINE
# ---------------------------- #
if __name__ == "__main__":
    template_path = "pictures/template.png"
    filled_doc_path = "TEST4.jpg"

    try:
        print("\n--- Step 1: Skew Correction ---")
        step1 = correct_skew(filled_doc_path, "1_skewed.png")

        print("\n--- Step 2: SIFT Alignment ---")
        step2 = align_documents_sift(template_path, step1, "2_sift.png")

        print("\n--- Step 3: Perspective Correction ---")
        step3 = correct_perspective(step2, "3_corrTab.png")

        print("\n--- Step 4: Remove Red Lines ---")
        step4 = remove_colored_lines(step3, "4_no_red.png")

        print("\n--- Step 5: Enhance Handwriting Final ---")
        step5 = enhance_handwriting_final(step4, "final_clean_strokes_white_on_black.png")

        print("\n--- Step 6: Extract Valid Letter Boxes ---")
        extract_valid_letter_boxes(step5, "valid_letter_boxes")

        print("\n✅ Pipeline complete — final outputs in 'valid_letter_boxes/'.")

        # =====================================================
        # LOOP THROUGH ALL IMAGES IN valid_letter_boxes FOLDER
        # =====================================================

        input_folder = r"C:\Users\cochi\OneDrive\Documents\OpenCV\valid_letter_boxes"
        output_folder = r"C:\Users\cochi\OneDrive\Documents\OpenCV\isolated_characters"
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f"isolated_{filename}")
                isolate_character_and_measure(input_path, output_path)

        print("\n✅ Finished processing all images in valid_letter_boxes.")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        
import cv2 as cv
import numpy as np
import os

# --- Input and Output directories ---
input_dir = "isolated_characters"
output_dir = "enhanced_characters"

# Create output folder if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# --- Loop through all images in isolated_characters ---
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # --- Load grayscale image ---
        img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipping {filename} (cannot read image).")
            continue

        # --- Step 1: Adaptive threshold for better contrast and binarization ---
        thresh = cv.adaptiveThreshold(img, 255,
                                      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, 15, 3)

        # --- Step 2: Use a smaller kernel to prevent over-thickening ---
        kernel = np.ones((4, 4), np.uint8)

        # --- Step 3: Controlled closing to connect gaps but preserve shape ---
        connected = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

        # --- Step 4: Gentle dilation (optional, fine-tune) ---
        dilated = cv.dilate(connected, np.ones((3, 3), np.uint8), iterations=1)

        # --- Step 5: Small median blur to smooth jagged pixels ---
        final = cv.medianBlur(dilated, 3)

        # --- Step 6: Optional sharpening to restore edge clarity ---
        sharp_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        final = cv.filter2D(final, -1, sharp_kernel)

        # --- Save the final enhanced image ---
        cv.imwrite(output_path, final)
        print(f"✅ Enhanced and saved: {output_path}")

print("\n🎉 All images processed and saved in 'enhanced_characters' folder!")
