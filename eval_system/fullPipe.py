import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf

from box_man import Boxman, Letter

from math import ceil, floor
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
from skimage.filters import threshold_sauvola

from pathlib import Path

# Typings
from numpy.typing import NDArray
from cv2.typing import MatLike

# keras
from keras.models import load_model

# Temporary set tensorflow to use cpu only
# tf.config.set_visible_devices([], 'GPU')

model_path = (
    "../letara_site/letara/model/handwriting_MNIST.keras"
)

model_path_lc = (
    "../letara_site/letara/model/hwv1.keras"
)

# model_path = 'letara'
loaded_model = load_model(model_path)
loaded_model_lcuc = load_model(model_path_lc)
# loaded_model.load_weights("./model/handwriting_MNIST.keras")

word_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',
    15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'
}

def box_lut(char_set: str) -> str:
    match char_set:
        case "efghijkl":
            return "e_to_l"
        case "mnopqrst":
            return "m_to_t"
        case "uvwxyz":
            return "u_to_z"
        case _:
            return "all_caps"

def plot_imgs(imgs: list, n_row, n_col, file_name=''):
    # plt.close('all')  # Close any existing figures FIRST
    
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    
    for img, ax in zip(imgs, axs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Optional: hide axes for cleaner look
    
    # Hide unused subplots
    for ax in axs[len(imgs):]:
        ax.axis('off')
    
    plt.tight_layout()
    if file_name.strip() != '': 
        plt.savefig(f'{file_name}.png', bbox_inches='tight')
    # plt.show()
    # plt.close('all')


def show_img(img: MatLike, title: str = '') -> None:
    plt.imshow(img, 'gray')
    plt.title(title)
    plt.show()

# accepts isolated character image returns a 28x28 img format same as model trained input
# with black bg and white fg
def preproc_char(img: MatLike, type=''):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = cv.bitwise_not(img)
    # img = clahe_binarization(img)
    if type.strip() == 'hw':
        kernel = np.ones((10,10),np.uint8)
        img = cv.dilate(img,kernel,iterations = 1)
    img = cv.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    
    # Add channel dimension (if model expects 1 channel)
    formatted_img = np.expand_dims(img, axis=-1)  # shape: (30, 30, 1)
    
    # Add batch dimension
    formatted_img = np.expand_dims(formatted_img, axis=0)   # shape: (1, 30, 30, 1)

    # use img for plotting; formatted_img for OCR evaluation
    return img, formatted_img


def template_char_check(img: MatLike):
    
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    pre_iso = preproc_char_iso(img)
    # show_img(img, 'img @ template_check')
    # invert colors
    img = cv.bitwise_not(pre_iso)
    
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape
    
    true_x = width
    true_y = height
    true_x2 = 0
    true_y2 = 0
    
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    x,y,w,h = 0,0,0,0
    for cntr in contours:
        x,y,w,h = cv.boundingRect(cntr)
        
        if cv.contourArea(cntr) < 300:
            continue
        # cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 5)

        
        true_x = min(true_x, x)
        true_y = min(true_y, y)

        true_x2 = max(true_x2, x + w)
        true_y2 = max(true_y2, y + h)
        # cv.rectangle(dup, (x, y), (x+w, y+h), (255, 255, 255), 10)

        # print(f'contourArea: {cv.contourArea(cntr)}')
        # print("x,y,w,h:",x,y,w,h)
    # current threshold for small letters is @ true_h <= 60
    # dup = img.copy()
    true_w = true_x2 - true_x
    true_h = true_y2 - true_y
    # print(f'len contours: {len(contours)}')
    # cv.rectangle(dup, (true_x, true_y), (true_x+true_w, true_y+true_h), (255, 255, 255), 5)
    # show_img(dup, 'img @ char_check')
    # print("x,y,w,h:",true_x,true_y,true_w,true_h)
    # print(f"returns {true_h}, {bottom}")

    cropped_to_bbox = img[true_y:true_y+true_h, true_x:true_x+true_w]
    pad = 10
    cropped = cv.copyMakeBorder(cropped_to_bbox, pad, pad, pad, pad, 
                            cv.BORDER_CONSTANT, value=0)
    # print(f'true_h: {true_h}')
    img = cv.resize(cropped, (28, 28))

    img_predict = img_format(img)

    if true_h > 60:
        prediction = loaded_model.predict(img_predict)       
        return img, word_dict[np.argmax(prediction)]
    else:
        prediction = loaded_model_lcuc.predict(img_predict)  
        return img, word_dict[np.argmax(prediction)].lower()
    # modify this such that if there is already an implementation 

'''
Red lines removal related functions
'''
def detect_red_flexible(image: MatLike, h_thresh=10, s_thresh=50, v_min=50):
    """
    Flexible red detection with separate thresholds
    
    Parameters:
    - h_thresh: Hue threshold (how much red vs orange/pink)
    - s_thresh: Saturation threshold (how vivid)
    - v_min: Minimum value/brightness (lower = include darker reds)
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Red range 1 (0-10)
    lower_red1 = np.array([0, s_thresh, v_min])
    upper_red1 = np.array([h_thresh, 255, 255])
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    
    # Red range 2 (170-180)
    lower_red2 = np.array([180 - h_thresh, s_thresh, v_min])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    
    # Combine
    mask = cv.bitwise_or(mask1, mask2)
    result = cv.bitwise_and(image, image, mask=mask)
    
    return result, mask

def white_mask_then_inpaint(image, mask, dilate_iterations=2, inpaint_radius=5, method='telea'):
    """
    First replace red with white, then inpaint for smoother results
    
    Parameters:
    - image: Original image
    - mask: Binary mask of red areas (255 = red, 0 = not red)
    - dilate_iterations: How much to expand the mask
    - inpaint_radius: Radius for inpainting
    - method: 'telea' or 'ns' (Navier-Stokes)
    """
    # Step 1: Dilate mask to ensure complete coverage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv.dilate(mask, kernel, iterations=dilate_iterations)
    
    
    # Step 2: Replace masked area with white
    white_replaced = image.copy()
    white_replaced[dilated_mask > 0] = [255, 255, 255]
    
    
    # Step 3: Inpaint to smooth the edges
    if method == 'telea':
        inpainted = cv.inpaint(white_replaced, dilated_mask, inpaint_radius, cv.INPAINT_TELEA)
    else:  # 'ns'
        inpainted = cv.inpaint(white_replaced, dilated_mask, inpaint_radius, cv.INPAINT_NS)
    
    return inpainted, white_replaced



# checks what page or what worksheet set
# Assumes that the worksheet
def check_page(img: MatLike):
    # ROW_SIZE = 6
    # COL2_RANGE = 11
    coords = Boxman(mode='check').cells
    # coords = fullPipe.box_man.Boxman().cells[:2]
    # print(f'coords{len(coords)}: {coords[0]} {coords[1]}')
    # show_img(img, 'check page img')
    #add list of character ranges for each page to return the correct file path or whatever representation
    max_hits = 0
    prob_set = ""
    initial_char_lc = False

    char_set = ["ABCDEFGHIJ",
                "KLMNOPQRST",
                "UVWXYZabcd",   # probably gonna have to change lowercase depending on how the model handles it
                "efghijkl",
                "mnopqrst",
                "uvwxyz",
                ]


    # first_chars = []
    print(f'check_page len coords: {len(coords)}')
    
    first_chars = []

    for chars in char_set:
        match chars:
            case "efghijkl":
                coords = Boxman("check_e_to_l").cells
            case "mnopqrst":
                coords = Boxman("check_m_to_t").cells
            case "uvwxyz":
                coords = Boxman("check_u_to_z").cells
        first_chars = []
        img_list = []
        curr_set = chars
        hits = 0
        print(f'chars: {chars}')
        for i in range(0, len(chars)):
            print(f'len(chars): {len(chars)}  |  len(coords): {len(coords)}')
            template_char = img[coords[i].y:coords[i].y+coords[i].h,coords[i].x:coords[i].x+coords[i].w]
            char_img, template_txt = template_char_check(template_char)
            if initial_char_lc:
                template_txt = template_txt.lower()
            if template_txt.lower() == chars[i].lower():
                hits += 1
            img_list.append(char_img)
            first_chars.append(template_txt)
            if template_txt.islower():
                initial_char_lc = True
            if (hits > max_hits):
                print(f'max_hits: {max_hits} | set: {prob_set}')
                max_hits = hits
                prob_set = curr_set

        print(f'chars: first_chars')
        print(f'{chars}: {first_chars}')
                
    # debugging purposes
    # for i in range (0, len(coords)): # iterate only till 2nd row
    #     template_char = img[coords[i].y:coords[i].y+coords[i].h,coords[i].x:coords[i].x+coords[i].w]
    #     char_img, template_txt = template_char_check(template_char)
    #     img_list.append(char_img)
        # first_chars.append(template_txt)
        
    # plt.close('all')  # Close before plotting
    # plot_imgs(img_list, 1, len(img_list), 'prob_set')
    # first_chars = ''.join(first_chars)
    # print(f'template_txt {first_chars}')
    
    # for chars in char_set:
    #     if chars[0:2].lower() == first_chars.lower():
    #         print(f"Char set found: {chars}")
    #         return chars
    # print("Char set not found")
    first_chars = ''.join(first_chars)
    print(f'first_chars: {first_chars}')
    print(f'prob_set: {prob_set}')
    return prob_set


def init_boxes(mode='all_caps') -> Boxman:
    print(f'selected mode: {mode}')
    boxes = Boxman(mode)
    # boxes.print_all()
    return boxes

# preprocess isolated charcter
def preproc_char_iso(img: MatLike, type=''):
    # show_img(img, 'preproc_char_iso input')
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Denoise
    img = cv.fastNlMeansDenoising(img, h=10)
    
    # Sauvola thresholding
    window_size = 25
    thresh_sauvola = threshold_sauvola(img, window_size=window_size, k=0.1) 
    img = img > thresh_sauvola
    
    # Convert boolean to uint8 (0 and 255)
    img = img.astype(np.uint8) * 255
    
    # Invert so character is white on black
    img = cv.bitwise_not(img)

    # dilate expects white fg and black bg
    if type.strip() == 'hw':
        kernel = np.ones((9, 9), np.uint8)
        img = cv.dilate(img, kernel, iterations=1)
        
        kernel = np.ones((3, 3), np.uint8)
        # Apply erosion
        img = cv.erode(img, kernel, iterations=1)
    
    # Crop to bounding box
    # coords = cv.findNonZero(img)
    # if coords is not None:
    #     x, y, w, h = cv.boundingRect(coords)
    #     img = img[y:y+h, x:x+w]
    
    # Add padding
    # pad = 10
    # img = cv.copyMakeBorder(img, pad, pad, pad, pad, 
    #                         cv.BORDER_CONSTANT, value=0)
    
    # Resize to square
    # img = cv.resize(img, (28, 28))
    
    # invert again to get white bg and black fg
    img = cv.bitwise_not(img)
    # show_img(img, 'preproc_char_iso')
    
    return img

def img_format(img: MatLike) -> MatLike: # formatting for character recoginition
    img = img.astype('float32') / 255.0 
    # Add channel dimension (if model expects 1 channel)
    formatted_img = np.expand_dims(img, axis=-1)  # shape: (30, 30, 1)
    
    # Add batch dimension
    formatted_img = np.expand_dims(formatted_img, axis=0)   # shape: (1, 30, 30, 1)

    return formatted_img

def count_grid_cells(img) -> int:
    """
    Automatically detects and counts grid cells in the image.
    Returns the number of cells found.
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Apply threshold to detect grid lines
    # _, binary = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV)

    binary = cv.adaptiveThreshold(
        img_gray, 
        255, 
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, 
        15, 
        10
    )
    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    cell_count = 0
    img_area = img_gray.shape[0] * img_gray.shape[1]
    
    for contour in contours:
        area = cv.contourArea(contour)
        
        # Filter: cells should be reasonably sized (not too small, not the whole image)
        if area > 1000 and area < img_area * 0.5:
            # Approximate the contour to see if it's rectangular
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)
            
            # If it has 4 corners (rectangle)
            if len(approx) == 4:
                x, y, w, h = cv.boundingRect(contour)
                # Check aspect ratio is reasonable for a cell
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:  
                    cell_count += 1
    
    print(f"Number of grid cells detected: {cell_count}")
    return cell_count

# Grading
# accepts isolated char image with black fg and white bg
# returns skewed corrected for grading
def eval_orientation(img):
    # WHITE = [255,255,255]
    # PADDING = 0
    # constant = cv.copyMakeBorder(img,PADDING,PADDING,PADDING,PADDING,cv.BORDER_CONSTANT,value=WHITE)
    # img_hdup = cv.hconcat([img, img])
    # imgs = []
    # imgs.append(img)
    angle = get_angle(img)
    deskewed_img = rotate(img, angle, border_value=(255, 255, 255))
    # border_value=(255, 255, 255)
    # imgs.append(deskewed_img)
    # plot_imgs(imgs,1,2)
    print(f"angle: {angle}")
    return angle, deskewed_img

# function returns confidence value for the expected letter and predicted letter of the model
def eval_letter_form(img, expected_char):
    # full_char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    full_char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    fcs_lc = "abcdefghijklmnopqrstuvwxyz"
    print(f'expected char: {expected_char}')
    char_idx = full_char_set.find(expected_char)
    if char_idx == -1:
        char_idx = fcs_lc.find(expected_char)
    
    prediction1 = loaded_model.predict(img)
    # if (char_idx == -1):
    prediction2 = loaded_model_lcuc.predict(img)
    print(f'prediction1: {word_dict[np.argmax(prediction1)]} \t | prediction2: {word_dict[np.argmax(prediction2)]}')
    print(f'cond {word_dict[np.argmax(prediction1)].lower()} == {word_dict[char_idx].lower()}')
    if word_dict[np.argmax(prediction1)].lower() == word_dict[char_idx].lower():
        print('matched prediction1')
        prediction = prediction1
    else:
        prediction = prediction2

    print(f"predicted char: {word_dict[np.argmax(prediction)]}; actual char: {word_dict[char_idx]}")
    # print(f"predicted char: {word_dict[np.argmax(prediction)]}; actual char: {word_dict[char_idx]}")
    print(f"conf_predicted: {prediction[0][np.argmax(prediction)]:.2f}; actual_conf: {prediction[0][char_idx]:.2f}")

    return prediction[0][char_idx], word_dict[np.argmax(prediction)]

# expects an inverted image black background white foreground
def eval_size_align(img):
    
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape
    dup = img.copy()
    true_x = width
    true_y = height
    true_x2 = 0
    true_y2 = 0
    # img = cv.bitwise_not(img) 
    
    # img = cv.threshold(img,128,255,cv.THRESH_BINARY_INV)[1]
    
    # Convert to 8-bit grayscale if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # If normalized float (0-1)
        # img = img.astype(np.uint8)  # If already in 0-255 range
        
    # Ensure it's single channel
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # thresh = cv.threshold(img,128,255,cv.THRESH_BINARY)[1]
    
    
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv.boundingRect(cntr)
        if cv.contourArea(cntr) < 200:
            continue
        true_x = min(true_x, x)
        true_y = min(true_y, y)

        true_x2 = max(true_x2, x + w)
        true_y2 = max(true_y2, y + h)
        # cv.rectangle(dup, (x, y), (x+w, y+h), (255, 255, 255), 10)

        print(f'contourArea: {cv.contourArea(cntr)}')
        # print("x,y,w,h:",x,y,w,h)
    true_w = true_x2 - true_x
    true_h = true_y2 - true_y
    bottom = height - true_y2 # distance from bottom
    top = height - true_y
    print(f'len contours: {len(contours)}')
    # cv.rectangle(dup, (true_x, true_y), (true_x+true_w, true_y+true_h), (255, 255, 255), 10)
    print("x,y,w,h:",true_x,true_y,true_w,true_h)
    # print(f"returns {true_h}, {bottom}")

    cropped_to_bbox = img[true_y:true_y+true_h, true_x:true_x+true_w]
    pad = 10
    cropped = cv.copyMakeBorder(cropped_to_bbox, pad, pad, pad, pad, 
                            cv.BORDER_CONSTANT, value=0)
    # show_img(dup, 'size')
    # returns size, align
    return true_h, bottom, cropped

def percentage_diff(n1, n2, eps=1e-8):
    denom = abs(n1 + n2) / 2
    if denom < eps:
        return 0
    return abs(n1 - n2) / denom * 100

def percent_error(n1, n2):
    """
   Computes percent error between numbers, this function will only be used in alignment grading

   Parameters:
   - n1: handwritten grid align
   - n2: template bottom grid align
    """
    # if n1 < n2: # have to check this assumption if hw align is lower than tmemplate
    #     return 100
    # else:
    if n1 == 0 and n2 == 0:
        return 0
    # Workaround in cases where template align is 0
    elif n2 == 0:
        n1 += 1
        n2 += 1
    return abs((n1 - n2) / n2) * 100

# base 60 Transmutation table used by deped
def transmute_grade(initial_grade):
    """
    Transmute an initial grade to a transmuted grade based on the conversion table.
    
    Args:
        initial_grade (float): The initial grade (0-100)
    
    Returns:
        float: The transmuted grade (60-100)
    """
    initial_grade = round(initial_grade, 2)
    # Grade conversion table: (min_initial, max_initial, transmuted)
    grade_table = [
        (100, 100, 100),
        (98.40, 99.99, 99),
        (96.80, 98.39, 98),
        (95.20, 96.79, 97),
        (93.60, 95.19, 96),
        (92.00, 93.59, 95),
        (90.40, 91.99, 94),
        (88.80, 90.39, 93),
        (87.20, 88.79, 92),
        (85.60, 87.19, 91),
        (84.00, 85.59, 90),
        (82.40, 83.99, 89),
        (80.80, 82.39, 88),
        (79.20, 80.79, 87),
        (77.60, 79.19, 86),
        (76.00, 77.59, 85),
        (74.40, 75.99, 84),
        (72.80, 74.39, 83),
        (71.20, 72.79, 82),
        (69.60, 71.19, 81),
        (68.00, 69.59, 80),
        (66.40, 67.99, 79),
        (64.80, 66.39, 78),
        (63.20, 64.79, 77),
        (61.60, 63.19, 76),
        (60.00, 61.59, 75),
        (56.00, 59.99, 74),
        (52.00, 55.99, 73),
        (48.00, 51.99, 72),
        (44.00, 47.99, 71),
        (40.00, 43.99, 70),
        (36.00, 39.99, 69),
        (32.00, 35.99, 68),
        (28.00, 31.99, 67),
        (24.00, 27.99, 66),
        (20.00, 23.99, 65),
        (16.00, 19.99, 64),
        (12.00, 15.99, 63),
        (8.00, 11.99, 62),
        (4.00, 7.99, 61),
        (0, 3.99, 60),
    ]
    
    # Find the appropriate transmuted grade
    for min_grade, max_grade, transmuted in grade_table:
        if min_grade <= initial_grade <= max_grade:
            return transmuted
    
    # Handle edge cases
    if initial_grade > 100:
        return 100
    elif initial_grade < 0:
        return 60
    
    return None

# Final evaluation following the criteria
def eval_char_final(letter: Letter, template_letter: Letter):
    # Transmutation table
    letter.letter_g     = (letter.letter_form * 100)
    letter.size_g       = abs(100 - percentage_diff(letter.size, template_letter.size))
    letter.line_align_g = abs(100 - percent_error(letter.line_align, template_letter.line_align))

    # MAX_SKEW = 45 # max acceptable skew of a character
    # TRUE_HEIGHT = 90 # height of template characters (px) measured in photo software
    # match grade.strip().lower():
    #     case 'k':
    #         # print('Kinder rubric selected')
    #         letter_form_percent = 60
    #         line_align_percent = 40
    #         orientation_percent = 60
    #         size_percent = 50
    #     case '1':
    #         # print('Grade 1 rubric selected')
    #         letter_form_percent = 40
    #         line_align_percent = 20
    #         orientation_percent = 40
    #         size_percent = 30
    #     case '2':
    #         letter_form_percent = 20
    #         line_align_percent = 10
    #         orientation_percent = 20
    #         size_percent = 10   # changed from 5 to 10 despite rubric 
    #                             # due to parts of template may be removed when removing grid
            
    # Calculate individual statuses
    # letter.letter_form_status = (100 - letter.letter_form * 100 <= letter_form_percent)
    # or (percentage_diff(letter.letter_form, template_letter.letter_form) <= letter_form_percent)
    # letter.orientation_status = abs(letter.orientation) <= abs(template_letter.orientation) or (percentage_diff(letter.orientation, template_letter.orientation) <= orientation_percent)
    # letter.orientation_status = True
    # letter.orientation_status = (100 * (abs(letter.orientation) / MAX_SKEW)) <= orientation_percent
    # letter.size_status = (percentage_diff(letter.size, TRUE_HEIGHT) <= size_percent) or (percentage_diff(letter.size, template_letter.size) <= size_percent)
    # letter.line_align_status = abs(letter.line_align) <= abs(template_letter.line_align) or (percentage_diff(letter.line_align, template_letter.line_align) <= line_align_percent)
    
    # Combine into single grade with equal weighting
    # statuses = [
    #     letter.letter_form_status,
    #     letter.orientation_status,
    #     letter.size_status,
    #     letter.line_align_status
    # ]
    #
    # letter.overall_status = sum(statuses) / len(statuses) >= 0.5  # Returns True if 50%+ pass
    # letter.overall_status = letter.size > 0 and letter.overall_status

    

# resize character
def resize_cr(img: MatLike):
    return cv.resize(img, (28, 28))

# grid cell counter helper function
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2)))

def get_rects(bin):
    squares = []
    contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            if max_cos < 0.1:
                squares.append(cnt)
    return squares
    
def count_rect(img: MatLike) -> int:
    assert img is not None, "image not found"

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Preprocess
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    
    # Get uniform rect grids
    squares = get_rects(binary)
    
    print(f"Found {len(squares)} uniform square grids")
    
    # Visualize
    # output = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    # output = cv.drawContours( image, squares, -1, (0, 255, 0), 3 )

    return len(squares)


# def correct_skew(image: MatLike, output_path: str) -> MatLike:
#     # image = cv.imread(input_image_path)
#     if image is None:
#         raise FileNotFoundError(f"❌ Could not load {input_image_path} for skew correction.")
#     angle = get_angle(image)
#     corrected = rotate(image, angle)
#     # cv.imwrite(output_path, corrected)
#
#     print(f"✅ Skew correction done -> {output_path}, angle: {angle}")
#
#     return image


# ---------------------------- #
# Step 1: SIFT Alignment
# ---------------------------- #
def align_documents_sift(template_color: MatLike, filled_doc_color: MatLike, output_path: str) -> MatLike:
    # template_color = cv.imread(template_path)
    # filled_doc_color = cv.imread(filled_doc_path)
    # if template_color is None or filled_doc_color is None:
    #     raise ValueError("❌ Could not load images. Check file paths.")
    
    if len(filled_doc_color.shape) == 3:
        filled_doc_gray = cv.cvtColor(filled_doc_color, cv.COLOR_BGR2GRAY)
    else:
        filled_doc_gray = filled_doc_color.copy()

    # if len(template_color.shape) == 3:
    template_gray = cv.cvtColor(template_color, cv.COLOR_BGR2GRAY)
    # else:
    #     template_gray = template_color.copy()

    sift = cv.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(template_gray, None)
    kp2, desc2 = sift.detectAndCompute(filled_doc_gray, None)
    if desc1 is None or desc2 is None:
        raise ValueError("❌ Could not find enough features in images.")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good_matches) < 10:
        raise ValueError("❌ Not enough good matches found for alignment.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # homography, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

    # homography, _ = cv.findHomography(
    #     dst_pts, src_pts, 
    #     cv.USAC_MAGSAC,  
    #     ransacReprojThreshold=3.0,
    #     maxIters=5000
    # )

    homography, _ = cv.findHomography(
        dst_pts, src_pts, 
        cv.RANSAC, 
        5.0
    )
    if homography is None:
        raise ValueError("❌ Could not compute homography.")

    h, w, _ = template_color.shape
    aligned_image = cv.warpPerspective(filled_doc_color, homography, (w, h))
    cv.imwrite(output_path, aligned_image)
    print(f"✅ SIFT alignment done -> {output_path}")
    return aligned_image


# ---------------------------- #
# Step 2: Perspective Correction
# ---------------------------- #
# have to check the current image first for the existing letters or character set 
def correct_perspective(image: MatLike, output_path: str="3_corrTab.png") -> MatLike:
    
    # char_set = {"ABCDEFGHIJ": "./template/A-J.png",
    #             "KLMNOPQRST": "./template/K-T.png",
    #             "UVWXYZabcd": "./template/U-d.png",
    #             "efghijklmn": "./template/e-n.png",
    #             "opqrstuvwx": "./template/o-x.png",
    #             "yz": "./template/y-z.png"}
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255,
                                   cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY_INV, 15, 10)
    cnts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    corrected = image
    
    if cnts:
        c = max(cnts, key=cv.contourArea)
        epsilon = 0.02 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            src_points = np.float32([p[0] for p in approx])
            s = src_points.sum(axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = src_points[np.argmin(s)]
            rect[2] = src_points[np.argmax(s)]
            diff = np.diff(src_points, axis=1)
            rect[1] = src_points[np.argmin(diff)]
            rect[3] = src_points[np.argmax(diff)]
            
            # Calculate dimensions based on top edge
            top_width = np.linalg.norm(rect[1] - rect[0])
            left_height = np.linalg.norm(rect[3] - rect[0])
            right_height = np.linalg.norm(rect[2] - rect[1])
            avg_height = (left_height + right_height) / 2

            # 100px height per character
            # if num_enclosed <= 12: # if number of cells is less than 60 supposedly it is assumed it is yz worksheet
            # # have to fix this tho
            # # Scale to desired height
            #     scale = 200 / avg_height
            #     width = int(top_width * scale)
            #     height = 200
            # else:
            height, width = thresh.shape
            width_ratio = width / height
            height = 1000
            width = ceil(width_ratio * 1000)

            dst_points = np.float32([[0, 0], [width-1, 0],
                                     [width-1, height-1], [0, height-1]])
            M = cv.getPerspectiveTransform(rect, dst_points)
            corrected = cv.warpPerspective(image, M, (width, height))

            
                
            # Bottom points stay at same y-level, expanded horizontally
            dst_points = np.float32([[0, 0], [width-1, 0],
                                     [width-1, height-1], [0, height-1]])
            M = cv.getPerspectiveTransform(rect, dst_points)
            corrected = cv.warpPerspective(image, M, (width, height))
    
    cv.imwrite(output_path, corrected)
    
    print(f"✅ Perspective correction done -> {output_path}")
    return corrected

'''
A simple function that removes shadows from 
https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
'''

def remove_shadow(img: MatLike) -> MatLike:
    rgb_planes = cv.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 101)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)

    return result_norm


# ---------------------------- #
# Step 3: Remove Red Lines
# ---------------------------- #


#after SIFT call this
def remove_colored_lines(img_bgr: MatLike, output_path: str="done.png",
                         hsv_red_min_sat=20, lab_ab_thresh=4,
                         dilate_iters=3, inpaint_radius=1) -> MatLike:
    # img_bgr = cv.imread(input_image_path, cv.IMREAD_COLOR)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    lower_red1 = np.array([0, hsv_red_min_sat, 30], dtype=np.uint8)
    upper_red1 = np.array([25, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([155, hsv_red_min_sat, 30], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_hsv_red = cv.bitwise_or(
        cv.inRange(hsv, lower_red1, upper_red1),
        cv.inRange(hsv, lower_red2, upper_red2)
    )

    # lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB).astype(np.int16)
    # a, b = lab[:, :, 1] - 128, lab[:, :, 2] - 128
    # mask_lab = ((np.abs(a) > lab_ab_thresh) | (np.abs(b) > lab_ab_thresh)).astype(np.uint8) * 255

    combined_mask = mask_hsv_red
    # combined_mask = cv.bitwise_or(mask_hsv_red, mask_lab)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv.dilate(combined_mask, kernel, iterations=dilate_iters)
    inpainted = cv.inpaint(img_bgr, mask_dilated, inpaint_radius, cv.INPAINT_TELEA)
    gray = cv.cvtColor(inpainted, cv.COLOR_BGR2GRAY)
    cv.imwrite(output_path, gray)
    print(f"✅ Red lines removed -> {output_path}")
    return gray


# ---------------------------- #
# Step 4: Extract Valid Letter Boxes
# ---------------------------- #

def remove_grid(img: MatLike) -> MatLike:
    result = img.copy()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

# Remove horizontal lines
# this settings is for orig resolution of WS
# horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (60,1))
# vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,150))
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (120,1))
    remove_horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(remove_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(result, [c], -1, (255,255,255), 5)

# Remove vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,300))
    remove_vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv.findContours(remove_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(result, [c], -1, (255,255,255), 5)

    # cv.imwrite('./res_imgs/lines_removed.png', result)
    return result

# accepts perspective corrected image; no red line, grid removed
def eval_letters(img: MatLike, box, char_set):
    boxes = box.cells
    num_col = 6
    num_row = len(char_set)
    unprocessed_imgs = []
    orientation_fixed = []
    eval_preproc = []
    imgs = []
    for i in range (0, num_row):
        exp_char = char_set[i]
        char_conf_lvl = 0
        for j in range(0, num_col):
            chr_isolated = img[boxes[i * num_col + j].y:boxes[i*num_col+j].y+boxes[i*num_col+j].h,boxes[i*num_col+j].x:boxes[i*num_col+j].x+boxes[i*num_col+j].w]
            imgs.append(chr_isolated)   
            if j == 0:    
                chr_isolated_whitebg = preproc_char_iso(chr_isolated)
                new_letter = Letter(exp_char, boxes[i*num_col+j], is_template=True)
            else:
                chr_isolated_whitebg = preproc_char_iso(chr_isolated, type='hw') # dilate image
                new_letter = Letter(exp_char, boxes[i*num_col+j], is_template=False)
            orientation_angle, fixed_orientation_img = eval_orientation(chr_isolated)
            
            # Invert color then resize to 28x28
            chr_isolated = cv.bitwise_not(chr_isolated_whitebg)
            new_letter.size, new_letter.line_align, new_letter_img = eval_size_align(chr_isolated)
            chr_isolated = resize_cr(new_letter_img)
            char_pred = img_format(chr_isolated)
            orientation_fixed.append(fixed_orientation_img)
  
            conf_lvl, _ = eval_letter_form(char_pred, exp_char)
            new_letter.letter_form = conf_lvl 
            new_letter.orientation = orientation_angle
            char_conf_lvl = char_conf_lvl + conf_lvl
            eval_preproc.append(chr_isolated)
            if j != 0:
                eval_char_final(new_letter, box.letters[i * num_col]) # evaluate non template character here
            box.letters.append(new_letter)
        print(f'{exp_char} conf level avg = {char_conf_lvl / 6}')
    # for box in boxes:
    #     chr_isolated = img[box.y:box.y+box.h,box.x:box.x+box.w]
    #     chr_isolated = preproc_char
    #     imgs.append(chr_isolated)

    # plot_imgs(imgs, num_row, num_col)
    # plot_imgs(orientation_fixed, num_row, num_col)
    plot_imgs(eval_preproc, num_row, num_col, file_name='imgs')

    # expects warped fix image or 3_corrTab.png
def create_result(img: MatLike, letters, out_path: str = ""):
    result = img
    for letter in letters:
        # letter.print_coords()
        # print(f"isPass: {letter.isPass()}")
        box = letter.box
        x = box.x
        y = box.y
        w = box.w
        h = box.h
        if letter.is_template:
            # print("Cont..")
            continue
        if letter.isPass() and not letter.is_template:
            print("Letter passed")

            # print(f"x:{x}, y:{y}, w:{w}, h:{h}") 
            cv.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 15)
        else:
            cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 15)
    return result


# ---------------------------- #
# MAIN PIPELINE
# ---------------------------- #
if __name__ == "__main__":
    try:
        template_path = "./template/A-J.png"
        filled_doc_path = "./worksheets/A-J_size.jpg"

        template_img = cv.imread(template_path)
        ws_img = cv.imread(filled_doc_path)

        imgs = []

        boxes = init_boxes()

        print("\n--- SIFT Alignment ---")
        sift_aligned = align_documents_sift(template_img, ws_img, "2_sift.png")

        print("\n--- Shadow Removal ---")

        shadow_removed = remove_shadow(sift_aligned)
        cv.imwrite("removed_shadow.png", shadow_removed)

        # remove this update the code to parse the new template 
        num_enclosed = count_rect(shadow_removed)

        print("\n--- Perspective Correction ---")
        num_enclosed = count_rect(sift_aligned)
        perspective_corrected = correct_perspective(sift_aligned, num_enclosed, "3_corrTab.png")
        grid_removed = remove_grid(perspective_corrected)

        print("\n--- Remove Red Lines ---")
        result, mask = detect_red_flexible(grid_removed, h_thresh=10, s_thresh=25, v_min=70)
        image_processed, white_telea = white_mask_then_inpaint(grid_removed, mask, dilate_iterations=2, inpaint_radius=1, method='telea')

        char_set = check_page(image_processed)
        print(f"chars: {char_set}")

        print("\n--- Eval Letters ---")
        eval_letters(image_processed, boxes, char_set) 

        res_img = create_result(perspective_corrected, boxes.letters)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
# isolate_chars(image_processed, boxes.cells)

# cleaned_telea = clean_artifacts(inpainted1_telea)

# chr_isolated = img[box.y:box.y+box.h,box.x:box.x+box.w]
# clahe_binarization(inpainted1_telea[])

# print("\n --- Cycle Characters ---")
# cycle_characters(cleaned_telea, boxes)



# print("\n--- Enhance Handwriting Final ---")
# count_enclosed(red_removed2)
# step5 = enhance_handwriting_final(cleaned_telea, "final_clean_strokes_white_on_black.png")

# print("\n✅ Pipeline complete — final outputs in 'valid_letter_boxes/'.")

