"""
Letter-by-Letter Handwriting Grading System with Repetitions
Grades 5 repetitions of each letter (A-Z)
"""
import os
import cv2 as cv
import numpy as np

from PIL import Image
from numpy._core.numeric import full
from tensorflow import size

import fullPipe
import box_man

def grade_handwriting_by_letter(image_path):
    """
    Main grading function - Grades 5 repetitions of each letter
    
    Args:
        image_path: Path to the cropped worksheet image
        
    Returns:
        dict: Contains all letter instances, per-letter summaries, and overall summary
        {
            'letter_instances': [
                {
                    'letter': 'A',
                    'repetition_number': 1,  # 1-5
                    'position_in_worksheet': 1,  # Overall position
                    'letter_form': 85.5,
                    'size': 90.0,
                    'line_align': 78.5,
                    'orientation': 88.0,
                    'bbox_x': 100,
                    'bbox_y': 50,
                    'bbox_width': 45,
                    'bbox_height': 60,
                    'comments': 'Good formation'
                },
                # ... 5 repetitions of A, then 5 of B, etc.
            ],
            'letter_summaries': [
                {
                    'letter': 'A',
                    'avg_letter_form': 85.0,
                    'avg_size': 87.0,
                    'avg_line_align': 82.0,
                    'avg_orientation': 86.0,
                    'letter_average': 85.0,
                    'repetition_count': 5,
                    'best_score': 90.0,
                    'worst_score': 78.0,
                    'comments': 'Overall good A formation'
                },
                # ... one summary per unique letter
            ],
            'worksheet_summary': {
                'overall_letter_form': 85.0,
                'overall_size': 87.0,
                'overall_line_align': 82.0,
                'overall_orientation': 86.0,
                'overall_score': 85.0,
                'total_letters': 26,  # A-Z
                'total_repetitions': 130,  # 26 letters × 5 repetitions
                'grading_method': 'automatic',
                'comments': 'Overall feedback',
                'strengths': 'What student does well',
                'areas_for_improvement': 'What needs work'
            }
        }
    """

    # model_path = (
    #     Path(settings.BASE_DIR)
    #         / "letara"
    #         / "model"
    #         / "handwriting_MNIST.keras"
    # )
    num_row = 10
    NUM_COL = 6

    template_img = cv.imread('../template/A-J.png')
    ws_img = cv.imread(image_path)
    print(f'image_path: {image_path}')

    assert ws_img is not None, "Image is not found"

    # init box coords
    boxes = fullPipe.init_boxes()
    # if ws_img is None:
    #     raise ValueError("Could not load image")
    print("\n--- SIFT Alignment ---")
    sift_aligned = fullPipe.align_documents_sift(template_img, ws_img, "2_sift.png")

    print("\n--- Shadow Removal ---")
    shadow_removed = fullPiperemove_shadow(sift_aligned)
    cv.imwrite("removed_shadow.png", shadow_removed)

    # num_enclosed = count_rect(shadow_removed) # don't need this anymore

    print("\n--- Perspective Correction ---")
    perspective_corrected = correct_perspective(shadow_removed, "3_corrTab.png")



    grid_removed = fullPipe.remove_grid(perspective_corrected)
    print("\n--- Remove Red Lines ---")
    result2, mask2 = fullPipe.detect_red_flexible(grid_removed, h_thresh=10, s_thresh=25, v_min=70)
    inpainted2_telea, _ = fullPipe.white_mask_then_inpaint(grid_removed, mask2, dilate_iterations=2, inpaint_radius=3, method='telea')
    image_processed = inpainted2_telea
    cv.imwrite("./red_removed.png", image_processed)
# image_processed = fullPipe.remove_colored_lines(grid_removed, "4_no_red.png")
# image_processed = 

# add thresholding here before passing to eval_letters
    char_set = fullPipe.check_page(image_processed)
    print(f"chars: {char_set}")
    boxes = box_man.Boxman(fullPipe.box_lut(char_set))

    print("\n--- Eval Letters ---")
    fullPipe.eval_letters(image_processed, boxes, char_set) 

    img_out = perspective_corrected.copy()
    res_img = fullPipe.create_result(img_out, boxes.letters)

    # remove this update the code to parse the new template 
    # all_letters = detect_all_letter_instances(img)

    # Grade each instance
    letter_instances = []
    num_row = len(char_set)
    # position = 1
    print(f'==========grading_system_debug==========')
    for i in range(num_row):
        for j in range(NUM_COL):
            curr_idx = i*NUM_COL+j
            if j == 0 or boxes.letters[curr_idx].size <= 0:  # character is a template or cell is empty
                continue
            else:
                letter_instances.append(letter_to_data(boxes.letters[curr_idx], j))

    # for letter_data in all_letters:
    #     instance_grade = letter_to_data(
    #     )
    #     letter_instances.append(instance_grade)
    #     position += 1
    
    # Calculate summaries per letter
    letter_summaries = calculate_letter_summaries(letter_instances)
    
    # Calculate worksheet summary
    worksheet_summary = calculate_worksheet_summary(letter_summaries, letter_instances)
    
    dump_letters(boxes, num_row)

    return {
        'letter_instances': letter_instances,
        'letter_summaries': letter_summaries,
        'worksheet_summary': worksheet_summary
    }


def detect_all_letter_instances(img):
    """
    Detect all 130 letter instances (26 letters × 5 repetitions each)
    
    YOUR CODE HERE:
    - Detect all letters in order
    - Group by letter type (A, A, A, A, A, B, B, B, B, B, ...)
    - Track which repetition number (1-5) for each
    
    Returns:
        list: [
            {
                'image': cropped_letter_image,
                'letter': 'A',
                'repetition': 1,  # Which repetition (1-5)
                'bbox': (x, y, w, h)
            },
            ...
        ]
    """
    # Example placeholder implementation
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by position (left to right, top to bottom)
    contours = sorted(contours, key=lambda c: (cv.boundingRect(c)[1] // 100, cv.boundingRect(c)[0]))
    
    all_letters = []
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter_index = 0
    repetition = 1
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 100 or area > 10000:
            continue
        
        x, y, w, h = cv.boundingRect(cnt)
        letter_img = img[y:y+h, x:x+w]
        
        current_letter = letters[letter_index] if letter_index < 26 else 'Z'
        
        all_letters.append({
            'image': letter_img,
            'letter': current_letter,
            'repetition': repetition,
            'bbox': (x, y, w, h)
        })
        
        # Move to next repetition/letter
        repetition += 1
        if repetition > 5:
            repetition = 1
            letter_index += 1
    
    return all_letters


def letter_to_data(letter: box_man.Letter, repetition_num):
    """
    Grade a single letter instance
    
    Args:
        letter_img: Image of the letter
        letter_char: Which letter (A-Z)
        repetition_num: Which repetition (1-5)
        position: Overall position in worksheet
        bbox: Bounding box (x, y, w, h)
    """
    # repetition_num += 1 # 1 based indexing

    # letter_form = analyze_letter_shape(letter_img, letter_char)
    # size = analyze_letter_size(letter_img, bbox)
    # alignment = analyze_letter_alignment(letter_img, bbox)
    # orientation = analyze_letter_orientation(letter_img)
    #
    # comments = f"{letter_char} rep {repetition_num}: " + generate_instance_feedback(
    #     letter_form, size, alignment, orientation
    # )
    letter_form = letter.letter_form * 100
    print(f'char: {letter.char} \t | form: {letter.letter_form: .2f} \t | form * 100: {letter_form: .2f}')
    return {
        'letter': letter.char,
        'repetition_num': repetition_num,
        'letter_form': letter_form,
        'size': letter.size_g,
        'line_align': letter.line_align_g,
        # 'orientation': letter.orientation
    }
    
    return {
        'letter': letter_char,
        'repetition_number': repetition_num,
        'position_in_worksheet': position,
        'letter_form': letter_form,
        'size': size,
        'line_align': alignment,
        'orientation': orientation,
        'bbox_x': bbox[0],
        'bbox_y': bbox[1],
        'bbox_width': bbox[2],
        'bbox_height': bbox[3],
        'comments': comments
    }


def calculate_letter_summaries(letter_instances):
    """
    Calculate average scores for each unique letter across its 5 repetitions
    
    Args:
        letter_instances: List of all graded instances
    
    Returns:
        list: One summary dict per unique letter
    """
    # Group by letter
    letters_dict = {}
    for instance in letter_instances:
        letter = instance['letter']
        if letter not in letters_dict:
            letters_dict[letter] = []
        letters_dict[letter].append(instance)
    
    # Calculate averages
    summaries = []
    for letter, instances in letters_dict.items():
        if not instances:
            continue
        
        for i in instances:
            print(f'letter_form: {i['letter_form']}')
            print(f'size: {i['size']}')
            print(f'line_align: {i['line_align']}')
            # print(f'orientation: {i['orientation']}')


        avg_form = sum(i['letter_form'] for i in instances) / len(instances)
        avg_size = sum(i['size'] for i in instances) / len(instances)
        avg_align = sum(i['line_align'] for i in instances) / len(instances)
        # avg_orient = sum(i['orientation'] for i in instances) / len(instances)
        
        letter_avg = (avg_form + avg_size + avg_align) / 3
        # letter_avg = (avg_form + avg_size + avg_align + avg_orient) / 4
        
        # Get best and worst
        # scores = [(i['letter_form'] + i['size'] + i['line_align'] + i['orientation']) / 4 
        #           for i in instances]

        scores = [(i['letter_form'] + i['size'] + i['line_align'] ) / 3 
                  for i in instances]
        
        summaries.append({
            'letter': letter,
            'avg_letter_form': round(avg_form, 2),
            'avg_size': round(avg_size, 2),
            'avg_line_align': round(avg_align, 2),
            # 'avg_orientation': round(avg_orient, 2),
            'letter_average': round(letter_avg, 2),
            'repetition_count': len(instances),
            'best_score': round(max(scores), 2) if scores else None,
            'worst_score': round(min(scores), 2) if scores else None,
            'comments': generate_letter_summary_feedback(letter, letter_avg, len(instances))
        })
    
    return summaries


def calculate_worksheet_summary(letter_summaries, letter_instances):
    """Calculate overall worksheet summary"""
    if not letter_summaries:
        return default_worksheet_summary()
    
    # Average across all letters
    avg_form = sum(l['avg_letter_form'] for l in letter_summaries) / len(letter_summaries)
    avg_size = sum(l['avg_size'] for l in letter_summaries) / len(letter_summaries)
    avg_align = sum(l['avg_line_align'] for l in letter_summaries) / len(letter_summaries)
    # avg_orient = sum(l['avg_orientation'] for l in letter_summaries) / len(letter_summaries)
    
    # overall = (avg_form + avg_size + avg_align + avg_orient) / 4
    overall = (avg_form + avg_size + avg_align) / 3
    
    return {
        'overall_letter_form': round(avg_form, 2),
        'overall_size': round(avg_size, 2),
        'overall_line_align': round(avg_align, 2),
        # 'overall_orientation': round(avg_orient, 2),
        'overall_score': round(overall, 2),
        'total_letters': len(letter_summaries),
        'total_repetitions': len(letter_instances),
        'grading_method': 'automatic',
        'graded_by': 'AI Grading System v1.0',
        'comments': generate_worksheet_comments(overall, len(letter_summaries), len(letter_instances)),
        # 'strengths': identify_worksheet_strengths(avg_form, avg_size, avg_align, avg_orient),
        'strengths': identify_worksheet_strengths(avg_form, avg_size, avg_align),
        # 'areas_for_improvement': identify_worksheet_improvements(avg_form, avg_size, avg_align, avg_orient)
        'areas_for_improvement': identify_worksheet_improvements(avg_form, avg_size, avg_align)
    }


# Placeholder analysis functions (replace with your actual code)
def analyze_letter_shape(letter_img, letter_char):
    return 85.0

def analyze_letter_size(letter_img, bbox):
    return 87.0

def analyze_letter_alignment(letter_img, bbox):
    return 82.0

def analyze_letter_orientation(letter_img):
    return 86.0

# def generate_instance_feedback(form, size, align, orient):
def generate_instance_feedback(form, size, align):
    # avg = (form + size + align + orient) / 4
    avg = (form + size + align) / 3
    if avg >= 85:
        return "Excellent!"
    elif avg >= 70:
        return "Good work"
    else:
        return "Needs practice"

def generate_letter_summary_feedback(letter, avg_score, rep_count):
    if avg_score >= 85:
        return f"Excellent {letter} formation across all {rep_count} repetitions"
    elif avg_score >= 70:
        return f"Good {letter} with room for improvement"
    else:
        return f"Letter {letter} needs more practice"

def generate_worksheet_comments(score, unique_letters, total_reps):
    return f"Graded {unique_letters} letters ({total_reps} total instances). Overall score: {score:.1f}"

# def identify_worksheet_strengths(form, size, align, orient):
def identify_worksheet_strengths(form, size, align):
    strengths = []
    if form >= 80: strengths.append("letter formation")
    if size >= 80: strengths.append("sizing")
    if align >= 80: strengths.append("alignment")
    # if orient >= 80: strengths.append("orientation")
    return ", ".join(strengths) if strengths else "Keep practicing"

# def identify_worksheet_improvements(form, size, align, orient):
def identify_worksheet_improvements(form, size, align):
    improvements = []
    if form < 70: improvements.append("letter formation")
    if size < 70: improvements.append("sizing")
    if align < 70: improvements.append("alignment")
    # if orient < 70: improvements.append("orientation")
    return ", ".join(improvements) if improvements else "Maintain current level"

def default_worksheet_summary():
    return {
        'overall_letter_form': 0, 'overall_size': 0,
        'overall_line_align': 0, 
        # 'overall_orientation': 0,
        'overall_score': 0, 'total_letters': 0, 'total_repetitions': 0,
        'grading_method': 'automatic', 'graded_by': 'AI System',
        'comments': 'No letters detected', 'strengths': '', 'areas_for_improvement': ''
    }
    """
    Main grading function - Grades each letter individually
    
    Args:
        image_path: Path to the cropped worksheet image
        
    Returns:
        dict: Contains letter grades and overall summary
        {
            'letters': [
                {
                    'letter': 'A',
                    'letter_number': 1,
                    'letter_form': 85.5,
                    'size': 90.0,
                    'line_align': 78.5,
                    'orientation': 88.0,
                    'bbox_x': 100,
                    'bbox_y': 50,
                    'bbox_width': 45,
                    'bbox_height': 60,
                    'comments': 'Good letter formation'
                },
                # ... more letters
            ],
            'summary': {
                'overall_letter_form': 85.0,
                'overall_size': 87.0,
                'overall_line_align': 82.0,
                'overall_orientation': 86.0,
                'overall_score': 85.0,
                'grading_method': 'automatic',
                'graded_by': 'AI Grading System v1.0',
                'comments': 'Overall good handwriting...',
                'strengths': 'Consistent letter sizing...',
                'areas_for_improvement': 'Work on baseline alignment...'
            }
        }
    """
    
    # Load image
    img = cv.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    
    # ========================================
    # YOUR LETTER DETECTION CODE HERE
    # ========================================
    
    # Step 1: Detect individual letters
    letters_data = detect_letters(img)
    
    # Step 2: Grade each letter
    letter_grades = []
    for i, letter_data in enumerate(letters_data):
        letter_img = letter_data['image']
        letter_char = letter_data.get('letter', 'X')  # Detected letter character
        
        # Grade this specific letter
        letter_grade = grade_single_letter(
            letter_img, 
            letter_char, 
            i + 1,
            letter_data['bbox']
        )
        letter_grades.append(letter_grade)
    
    # Step 3: Calculate overall summary
    summary = calculate_summary(letter_grades)
    
    return {
        'letters': letter_grades,
        'summary': summary
    }

# Debugging purposes
def dump_letters(boxes, unique_letters):
    print(len(boxes.letters))
    written_acc = 0
    ROW_CNT = unique_letters
    COL_CNT = 6
    for i in range(ROW_CNT):
        for j in range(COL_CNT):
            if boxes.letters[i * COL_CNT + j].size > 0 and j != 0:
                curr_letter = boxes.letters[i * COL_CNT + j]
                print(f'char: {curr_letter.char}', end='\t')
                print(f'| idx: [{i}][{j}]\t | size: {curr_letter.size}')
                written_acc += 1

            
    print(written_acc)


def detect_letters(img):
    """
    Detect individual letters in the worksheet image
    
    YOUR CODE HERE:
    - Use contour detection to find letter boundaries
    - OCR to identify which letter (optional)
    - Return bounding boxes and letter images
    
    Returns:
        list of dicts: [
            {
                'image': cropped_letter_image,
                'letter': 'A',  # OCR detected letter
                'bbox': (x, y, width, height)
            },
            ...
        ]
    """
    # Placeholder implementation
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    letters_data = []
    for i, cnt in enumerate(contours):
        # Filter by area (adjust based on your worksheet)
        area = cv.contourArea(cnt)
        if area < 100 or area > 10000:  # Adjust these values
            continue
        
        x, y, w, h = cv.boundingRect(cnt)
        
        # Extract letter image
        letter_img = img[y:y+h, x:x+w]
        
        # Optional: Use OCR to detect letter
        # letter_char = detect_letter_with_ocr(letter_img)
        letter_char = 'X'  # Placeholder
        
        letters_data.append({
            'image': letter_img,
            'letter': letter_char,
            'bbox': (x, y, w, h)
        })
    
    return letters_data


def grade_single_letter(letter_img, letter_char, letter_number, bbox):
    """
    Grade a single letter
    
    YOUR GRADING CODE HERE:
    - Analyze letter formation
    - Check size
    - Measure alignment
    - Assess orientation
    
    Args:
        letter_img: Cropped image of the letter
        letter_char: The letter character (A-Z)
        letter_number: Position number
        bbox: (x, y, width, height) of letter
    
    Returns:
        dict: Letter grade data
    """
    
    # YOUR ANALYSIS CODE
    letter_form_score = analyze_letter_shape(letter_img, letter_char)
    size_score = analyze_letter_size(letter_img, bbox)
    alignment_score = analyze_letter_alignment(letter_img, bbox)
    # orientation_score = analyze_letter_orientation(letter_img)
    
    # Generate feedback
    comments = generate_letter_feedback(
        letter_char,
        letter_form_score,
        size_score,
        alignment_score,
        # orientation_score
    )
    
    return {
        'letter': letter_char,
        'letter_number': letter_number,
        'letter_form': letter_form_score,
        'size': size_score,
        'line_align': alignment_score,
        # 'orientation': orientation_score,
        'bbox_x': bbox[0],
        'bbox_y': bbox[1],
        'bbox_width': bbox[2],
        'bbox_height': bbox[3],
        'comments': comments
    }

if __name__ == "__main__":
    print("test")

