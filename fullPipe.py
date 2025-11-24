import cv2
import numpy as np
import os
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate

# ---------------------------- #
# Step 1: Skew Correction
# ---------------------------- #
def correct_skew(input_image_path, output_path):
    """Corrects the skew of an image."""
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
    """Aligns a filled document to a template using SIFT."""
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
    if len(desc1) < 2 or len(desc2) < 2:
        raise ValueError(f"❌ Not enough descriptors found. Need at least 2.")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Check for valid matches
    if not matches or (len(matches) > 0 and len(matches[0]) < 2):
         raise ValueError("❌ FLANN did not return k=2 matches. Check image quality.")

    # --- FIX IS HERE ---
    # Relaxed the matching criteria to be more forgiving
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance] # Was 0.7
    
    # Lowered the minimum number of matches required
    if len(good_matches) < 4: # Was 10
        raise ValueError(f"❌ Not enough good matches found ({len(good_matches)}). Need at least 4.")
    # --- END FIX ---

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
    """Corrects perspective distortion by finding the largest quad."""
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
    """Removes colored (e.g., red) lines using HSV and LAB color spaces."""
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
def enhance_handwriting_final(image_path: str, output_path: str, morph_kernel_size: int = 3):
    """
    Applies aggressive enhancement to the cleaned image to achieve
    crisp, white-on-black strokes.
    """
    print("\n--- Step 5: Enhancing Handwriting to White-on-Black Strokes ---")
    try:
        # --- FIX IS HERE ---
        # Load the image *directly* as grayscale, since Step 4 saved it that way.
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if gray is None:
            raise FileNotFoundError(f"❌ Could not load image at {image_path}")
        # --- END FIX ---

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
    """Extracts bounding boxes of potential characters based on size and density."""
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
    return output_dir

# ---------------------------- #
# Step 7: Isolate Character
# ---------------------------- #
def isolate_character_and_measure(image_path, output_path="isolated_character_output.png"):
    """
    Isolates the character in the image (assuming it's a feature within a frame),
    measures its dimensions, and saves the cropped result.
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: Could not load image at '{image_path}'.")
        return

    _, binary_inverted = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No significant contours found in {image_path}. Check threshold.")
        return

    contour_areas = [(cv2.contourArea(c), c) for c in contours]
    min_area = 100
    valid_contours = [item for item in contour_areas if item[0] > min_area]

    if not valid_contours:
        print(f"No character contours found after filtering {image_path}.")
        return
        
    valid_contours.sort(key=lambda x: x[0], reverse=True)

    if len(valid_contours) > 1:
        char_contours = [item[1] for item in valid_contours[1:]]  # skip biggest (frame)
    else:
        char_contours = [valid_contours[0][1]] # Assume only char, no frame

    all_points = np.vstack(char_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    padding = 5
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img_gray.shape[1], x + w + padding)
    y_end = min(img_gray.shape[0], y + h + padding)

    isolated_char = img_gray[y_start:y_end, x_start:x_end]
    
    print(f"--- Measured: {w} x {h} pixels for {os.path.basename(image_path)} ---")
    cv2.imwrite(output_path, isolated_char)
    return output_path

# ---------------------------- #
# Step 8: Final Character Enhancement
# ---------------------------- #
def enhance_final_character(input_path, output_path):
    """Applies final morphology and sharpening to an isolated character."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Skipping {input_path} (cannot read image).")
        return

    thresh = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 3)
    
    kernel = np.ones((4, 4), np.uint8)
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(connected, np.ones((3, 3), np.uint8), iterations=1)
    final = cv2.medianBlur(dilated, 3)

    sharp_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    final = cv2.filter2D(final, -1, sharp_kernel)
    
    cv2.imwrite(output_path, final)
    print(f"✅ Enhanced and saved: {output_path}")

# ---------------------------------------------------- #
# NEW FUNCTION: To wrap the loose code from original file
# ---------------------------------------------------- #
def enhance_all_isolated_characters(input_dir="isolated_characters", output_dir="enhanced_characters"):
    """
    Loops through all images in input_dir, enhances them,
    and saves them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Step 8: Enhancing All Isolated Characters ---")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            enhance_final_character(input_path, output_path)
            
    print(f"\n🎉 All images processed and saved in '{output_dir}' folder!")


# ---------------------------- #
# MAIN PIPELINE (for standalone running)
# ---------------------------- #
if __name__ == "__main__":
    
    # --- Configuration ---
    TEMPLATE_PATH = "pictures/template.png"
    INPUT_DOC_PATH = "TEST4.jpg"
    
    # --- Output Folders (using relative paths) ---
    DIR_STEP_6 = "valid_letter_boxes"
    DIR_STEP_7 = "isolated_characters"
    DIR_STEP_8 = "enhanced_characters"
    
    os.makedirs(DIR_STEP_6, exist_ok=True)
    os.makedirs(DIR_STEP_7, exist_ok=True)
    os.makedirs(DIR_STEP_8, exist_ok=True)

    try:
        # --- Main Pre-processing ---
        print("\n--- Step 1: Skew Correction ---")
        step1 = correct_skew(INPUT_DOC_PATH, "1_skewed.png")

        print("\n--- Step 2: SIFT Alignment ---")
        step2 = align_documents_sift(TEMPLATE_PATH, step1, "2_sift.png")

        print("\n--- Step 3: Perspective Correction ---")
        step3 = correct_perspective(step2, "3_corrTab.png")

        print("\n--- Step 4: Remove Red Lines ---")
        step4 = remove_colored_lines(step3, "4_no_red.png")

        print("\n--- Step 5: Enhance Handwriting Final ---")
        step5 = enhance_handwriting_final(step4, "final_clean_strokes_white_on_black.png")

        print("\n--- Step 6: Extract Valid Letter Boxes ---")
        step6_dir = extract_valid_letter_boxes(step5, DIR_STEP_6)

        print("\n✅ Pre-processing complete — final outputs in 'valid_letter_boxes/'.")

        # =====================================================
        # LOOP 1: Isolate characters from boxes
        # =====================================================
        print(f"\n--- Step 7: Isolating Characters from '{step6_dir}' ---")
        for filename in os.listdir(step6_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(step6_dir, filename)
                output_path = os.path.join(DIR_STEP_7, f"isolated_{filename}")
                isolate_character_and_measure(input_path, output_path)
        print(f"\n✅ Finished isolating characters in '{DIR_STEP_7}'.")

        # =====================================================
        # LOOP 2: Enhance isolated characters
        # =====================================================
        # This now calls the new function
        enhance_all_isolated_characters(DIR_STEP_7, DIR_STEP_8)

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")