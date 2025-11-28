from json import load
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
from nicegui import ui, run
from components.layout import layout
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter

import io
import traceback
import cv2 as cv
import numpy as np
import asyncio

# ---------------------------- #
# ▼▼▼▼▼▼▼▼▼▼▼ FIX 1 HERE ▼▼▼▼▼▼▼▼▼▼
# ---------------------------- #
# Add the project's root directory to the Python path
# This ensures the subprocess can find 'fullPipe.py'
import sys
ROOT_DIR = str(Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ---------------------------- #


# Import the pipeline functions from your fullPipe.py file
try:
    from fullPipe import (
        correct_skew,
        align_documents_sift,
        correct_perspective,
        remove_colored_lines,
        count_grid_cells,
        remove_grid,
        create_result,
        eval_letters,
        init_boxes,
        check_page,
        template_char_check
        # enhance_handwriting_final
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CRITICAL WARNING: Could not import fullPipe.py. {e}")
    PIPELINE_AVAILABLE = False


UPLOAD_DIR = Path('./data/uploads');  UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = Path('./data/results');  RESULT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_PATH = "./template/A-J.png" # Path to your template, relative to main.py

def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def _save_bytes_to_png(data: bytes, out_dir: Path, base: str) -> Path:
    img = Image.open(io.BytesIO(data)).convert('RGBA')
    out_path = out_dir / f'{base}.png'
    img.save(out_path, format='PNG')
    return out_path

# ---------------------------- #
# ▼▼▼▼▼▼▼▼▼▼▼ FIX 2 HERE ▼▼▼▼▼▼▼▼▼▼
# ---------------------------- #
# Restored the original function to catch *real* pipeline errors
# and show the fallback, rather than crashing the app.
def _make_result(in_path: Path) -> Path:
    """
    Runs the full pre-processing pipeline (Steps 1-5)
    to generate the single 'result' image for the UI.
    """
    if not PIPELINE_AVAILABLE:
        # Fallback to simple edge detect if import failed
        img = Image.open(in_path).convert('RGB').filter(ImageFilter.FIND_EDGES)
        out_path = RESULT_DIR / f'{in_path.stem}_result_fallback.png'
        img.save(out_path, format='PNG')
        return out_path

    print(f"--- Running full pipeline for {in_path.name} ---")
    
    # Define intermediate/final paths
    ts = in_path.stem # e.g., 'letara_20251114_133000'
    # step1_path = RESULT_DIR / f"{ts}_1_skewed.png"
    # step2_path = RESULT_DIR / f"{ts}_2_sift.png"
    # step3_path = RESULT_DIR / f"{ts}_3_corrTab.png"
    # step4_path = RESULT_DIR / f"{ts}_4_no_red.png"
    loaded_model = load_model('./model/handwriting_MNIST.keras')
    
    try:
        # Run the pipeline steps
        # s1 = correct_skew(str(in_path), str(step1_path))
        template_path = "./template/A-J.png"
        dummy_input = np.random.rand(1, 28, 28, 1).astype('float32')
        try:
            # dummy_output = await asyncio.to_thread(loaded_model.predict, dummy_input)
            dummy_output = asyncio.run(asyncio.to_thread(loaded_model.predict, dummy_input))
            # dummy_output = loaded_model.predict(dummy_input, verbose=0)
            print(f"Dummy prediction successful! Output shape: {dummy_output.shape}")
        except Exception as e:
            print(f"Error with dummy prediction: {e}")

        template_img = cv.imread(TEMPLATE_PATH)
        ws_img = cv.imread(str(in_path))

        boxes = init_boxes()
        # sift_aligned = align_documents_sift(template_img, ws_img, "2_sift.png")
        # s2 = align_documents_sift(TEMPLATE_PATH, s1, str(step2_path))
        # s3 = correct_perspective(s2, str(step3_path))
        # s4 = remove_colored_lines(s3, str(step4_path))
        # s5 = enhance_handwriting_final(s4, str(out_path))
        
        print("\n--- SIFT Alignment ---")
        sift_aligned = align_documents_sift(template_img, ws_img, "2_sift.png")
        red_removed = remove_colored_lines(sift_aligned, "4_no_red.png")

        num_enclosed = count_grid_cells(sift_aligned)

        print("\n--- Perspective Correction ---")
# perspective_corrected = correct_perspective(sift_aligned, "3_corrTab.png")
        perspective_corrected = correct_perspective(sift_aligned, num_enclosed, "3_corrTab.png")
        char_set = check_page(red_removed)
        print(f"chars: {char_set}")


        grid_removed = remove_grid(perspective_corrected)
        print("\n--- Remove Red Lines ---")
# show_img(sharpen, 'sharpened')
        image_processed = remove_colored_lines(grid_removed, "4_no_red.png")

        print("\n--- Eval Letters ---")
        eval_letters(image_processed, boxes, char_set) 

        res_img = create_result(perspective_corrected, boxes.letters, out_path)
        # show_img(res_img, 'result')
        print(f"--- Pipeline complete. Result: {out_path} ---")
        return Path(out_path) # Return the final path
    
    except Exception as e:
        print(f"❌ Pipeline failed for {in_path.name}: {e}")
        traceback.print_exc() # Print the *real* pipeline error to the console
        # As a fallback, just do the old edge detect so the UI doesn't break
        img = Image.open(in_path).convert('RGB').filter(ImageFilter.FIND_EDGES)
        fallback_path = RESULT_DIR / f'{in_path.stem}_fallback_error.png'
        img.save(fallback_path, format='PNG')
        ui.notify(f'Pipeline Error: {e}. Showing fallback.', color='negative')
        return fallback_path
# ---------------------------- #


async def _get_upload_bytes(e) -> bytes:
    """Robustly extract bytes for NiceGUI v3 upload event shapes."""
    if hasattr(e, 'content') and e.content is not None:
        try:
            return await e.content.read() 
        except Exception:
            try:
                await e.content.seek(0); return await e.content.read()
            except Exception:
                pass
    if hasattr(e, 'file') and e.file is not None:
        try:
            return await e.file.read()
        except Exception:
            try:
                await e.file.seek(0); return await e.file.read()
            except Exception:
                pass
    if hasattr(e, 'bytes') and e.bytes is not None:
        return e.bytes
    if hasattr(e, 'path') and e.path:
        return Path(e.path).read_bytes()
    raise ValueError('Could not read uploaded file bytes from event')

@ui.page('/upload')
def upload_page():
    layout('upload')
    ui.label('Image Upload & Results').classes('text-2xl font-bold mt-4 text-[#22c55e]')
    ui.label('Upload an image; we run the full pipeline and show the enhanced result.').classes('text-gray-200')

    with ui.row().classes('mt-4 gap-6 items-start flex-wrap'):
        # LEFT: uploader + paths
        with ui.column().classes('gap-3 w-[360px] max-w-full'):
            saved_path_in  = ui.input('Uploaded file path').props('readonly').classes('w-full')
            saved_path_out = ui.input('Result file path').props('readonly').classes('w-full')

            async def copy_to_clipboard(inp: ui.input):
                if inp.value:
                    await ui.run_javascript(f'navigator.clipboard.writeText({inp.value!r});')
                    ui.notify('Path copied to clipboard', color='info')
                else:
                    ui.notify('No path yet', color='warning')

            with ui.row().classes('gap-2'):
                ui.button('Copy upload path', on_click=lambda: copy_to_clipboard(saved_path_in)).props('color=secondary')
                ui.button('Copy result path', on_click=lambda: copy_to_clipboard(saved_path_out)).props('color=accent')

            async def on_upload(e):
                try:
                    data = await _get_upload_bytes(e)
                    base = f'letara_{_timestamp()}'
                    in_path  = _save_bytes_to_png(data, UPLOAD_DIR, base)
                    
                    # Notify user that processing has started (can take time)
                    ui.notify('Upload received. Running processing pipeline...', color='info')
                    
                    out_path = await run.cpu_bound(_make_result, in_path)
                    # ---------------------------------

                    saved_path_in.value  = str(in_path)
                    saved_path_out.value = str(out_path)

                    preview_in.set_source(str(in_path))
                    preview_out.set_source(str(out_path))
                    ui.notify('Pipeline processing complete ✓', color='positive')
                except Exception as ex:
                    ui.notify(f'Upload error: {ex}', color='negative')
                    print(f"❌ Upload failed: {ex}") # Also print to console
                    
            ui.upload(
                label='Select image…',
                on_upload=on_upload,
                auto_upload=True,
                multiple=False,
                max_files=1,
            ).props('accept="image/*"')

        # RIGHT: previews
        with ui.column().classes('gap-3'):
            ui.label('Preview').classes('font-semibold text-[#86efac]')
            preview_in  = ui.image().classes('rounded-xl border border-green-700 w-[420px] max-w-full')
            ui.label('Result (from Pipeline)').classes('font-semibold text-[#86efac]')
            preview_out = ui.image().classes('rounded-xl border border-green-700 w-[420px] max-w-full')
