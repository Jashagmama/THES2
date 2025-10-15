from nicegui import ui
from components.layout import layout
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter
import io

UPLOAD_DIR = Path('./data/uploads');  UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = Path('./data/results');  RESULT_DIR.mkdir(parents=True, exist_ok=True)

def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def _save_bytes_to_png(data: bytes, out_dir: Path, base: str) -> Path:
    img = Image.open(io.BytesIO(data)).convert('RGBA')
    out_path = out_dir / f'{base}.png'
    img.save(out_path, format='PNG')
    return out_path

def _make_result(in_path: Path) -> Path:
    """Simple preview result using Pillow (edge detect)."""
    img = Image.open(in_path).convert('RGB').filter(ImageFilter.FIND_EDGES)
    out_path = RESULT_DIR / f'{in_path.stem}_result.png'
    img.save(out_path, format='PNG')
    return out_path

def _get_upload_bytes(e) -> bytes:
    """Robustly extract bytes for NiceGUI v3 upload event shapes."""
    if hasattr(e, 'content') and e.content is not None:
        try:
            return e.content.read()
        except Exception:
            try:
                e.content.seek(0); return e.content.read()
            except Exception:
                pass
    if hasattr(e, 'file') and e.file is not None:
        try:
            return e.file.read()
        except Exception:
            try:
                e.file.seek(0); return e.file.read()
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
    ui.label('Upload an image; we save it and show a simple processed result.').classes('text-gray-200')

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
                    data = _get_upload_bytes(e)
                    base = f'letara_{_timestamp()}'
                    in_path  = _save_bytes_to_png(data, UPLOAD_DIR, base)
                    out_path = _make_result(in_path)

                    saved_path_in.value  = str(in_path)
                    saved_path_out.value = str(out_path)

                    preview_in.set_source(str(in_path))
                    preview_out.set_source(str(out_path))
                    ui.notify('Upload processed ✓', color='positive')
                except Exception as ex:
                    ui.notify(f'Upload error: {ex}', color='negative')

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
            ui.label('Result').classes('font-semibold text-[#86efac]')
            preview_out = ui.image().classes('rounded-xl border border-green-700 w-[420px] max-w-full')
