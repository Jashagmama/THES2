from django.shortcuts import render

# app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
from datetime import datetime
import base64, io, traceback
from PIL import Image, ImageFilter

# ---- directories ----
UPLOAD_DIR = Path('data/uploads')
RESULT_DIR = Path('data/results')
CAPTURE_DIR = Path('data/captures')

for d in [UPLOAD_DIR, RESULT_DIR, CAPTURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- HOME ----
def home(request):
    uploads = sorted(
        UPLOAD_DIR.glob('*.png'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:5]

    return render(request, 'home.html', {'uploads': uploads})


# ---- PROFILE ----
def profile(request):
    saved = False
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        about = request.POST.get('about')
        saved = True  # normally save to DB

    return render(request, 'profile.html', {'saved': saved})


# ---- CAMERA ----
@csrf_exempt
def save_camera_image(request):
    if request.method == 'POST':
        data_url = request.POST.get('image')
        header, encoded = data_url.split(',', 1)
        raw = base64.b64decode(encoded)

        img = Image.open(io.BytesIO(raw)).convert('RGBA')
        out = CAPTURE_DIR / f"letara_{datetime.now():%Y%m%d_%H%M%S}.png"
        img.save(out)

        return JsonResponse({'path': str(out)})

    return JsonResponse({'error': 'Invalid request'}, status=400)


def camera(request):
    return render(request, 'camera.html')


# ---- UPLOAD PIPELINE ----
def _timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _run_pipeline(in_path: Path) -> Path:
    try:
        from fullPipe import (
            correct_skew,
            align_documents_sift,
            correct_perspective,
            remove_colored_lines,
            enhance_handwriting_final
        )

        ts = in_path.stem
        step1 = RESULT_DIR / f"{ts}_1.png"
        step2 = RESULT_DIR / f"{ts}_2.png"
        step3 = RESULT_DIR / f"{ts}_3.png"
        step4 = RESULT_DIR / f"{ts}_4.png"
        final = RESULT_DIR / f"{ts}_final.png"

        s1 = correct_skew(str(in_path), str(step1))
        s2 = align_documents_sift("pictures/template.png", s1, str(step2))
        s3 = correct_perspective(s2, str(step3))
        s4 = remove_colored_lines(s3, str(step4))
        s5 = enhance_handwriting_final(s4, str(final))

        return Path(s5)

    except Exception:
        img = Image.open(in_path).filter(ImageFilter.FIND_EDGES)
        fallback = RESULT_DIR / f"{in_path.stem}_fallback.png"
        img.save(fallback)
        traceback.print_exc()
        return fallback


@csrf_exempt
def upload(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        base = f"letara_{_timestamp()}"
        in_path = UPLOAD_DIR / f"{base}.png"

        Image.open(img_file).save(in_path)
        out_path = _run_pipeline(in_path)

        context.update({
            'input_path': in_path,
            'output_path': out_path
        })

    return render(request, 'upload.html', context)

