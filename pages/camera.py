# LeTARA/pages/camera.py
from nicegui import ui
from components.layout import layout
from pathlib import Path
from datetime import datetime
import base64, io
from PIL import Image

SAVE_DIR = Path('./data/captures'); SAVE_DIR.mkdir(parents=True, exist_ok=True)

def _save_data_url_png(data_url: str) -> Path:
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    raw = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(raw)).convert('RGBA')
    out = SAVE_DIR / f'letara_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    img.save(out, format='PNG')
    return out

@ui.page('/camera')
def camera_page():
    layout('camera')
    ui.label('Camera').classes('text-2xl font-bold mt-4 text-[#22c55e]')
    ui.label('Click “Start Camera”, allow permission, then “Take Picture & Save”.').classes('text-gray-200')

    # --- Side-by-side layout (wraps on small screens) ---
    with ui.row().classes('mt-4 gap-6 items-start flex-wrap'):
        # LEFT: live preview
        ui.html(
            '<video id="letara_vid" autoplay playsinline '
            'style="width: 520px; max-width: 100%; border-radius: 12px; border: 2px solid #166534;"></video>',
            sanitize=False,
        )

        # RIGHT: controls + result + path
        with ui.column().classes('gap-3 w-[360px] max-w-full'):
            status_label = ui.label('').classes('text-gray-300')

            async def start_camera():
                try:
                    status = await ui.run_javascript('return letara_start_cam();', timeout=10.0)
                    if isinstance(status, str) and status.startswith('ok'):
                        status_label.set_text('Camera started ✓')
                        ui.notify('Camera ready', color='positive')
                    else:
                        status_label.set_text(str(status))
                        ui.notify(str(status) or 'Failed to start camera', color='negative')
                except Exception as e:
                    status_label.set_text(f'Error: {e}')
                    ui.notify(f'Error: {e}', color='negative')

            ui.button('Start Camera', on_click=start_camera).props('color=secondary')

            result_img = ui.image().classes('rounded-xl border border-green-600 w-full')
            path_out = ui.input('Saved file path').props('readonly').classes('w-full')

            async def capture_and_save():
                try:
                    data_url = await ui.run_javascript('return letara_capture_png();', timeout=10.0)
                    if not data_url:
                        ui.notify('No image captured', color='warning')
                        return
                    saved = _save_data_url_png(data_url)
                    result_img.set_source(str(saved))
                    path_out.value = str(saved)
                    ui.notify('📸 Saved to disk', color='positive')
                except Exception as e:
                    ui.notify(f'Capture error: {e}', color='negative')

            ui.button('Take Picture & Save', on_click=capture_and_save).props('color=positive')

            async def copy_path():
                if path_out.value:
                    await ui.run_javascript(f'navigator.clipboard.writeText({path_out.value!r});')
                    ui.notify('File path copied', color='info')
                else:
                    ui.notify('No file saved yet', color='warning')

            ui.button('Copy path', on_click=copy_path).props('color=accent')

    # Hidden canvas for snapshot; JS helpers
    ui.html('<canvas id="letara_can" style="display:none;"></canvas>', sanitize=False)
    ui.add_head_html('''
        <script>
        async function letara_start_cam() {
            try {
                const v = document.getElementById('letara_vid');
                const stream = await navigator.mediaDevices.getUserMedia({
                video:{
                facingMode: 'environment'
                }, audio: false 
                });
                v.srcObject = stream;
                return 'ok';
            } catch (e) {
                return 'error: ' + e;
            }
        }
        function letara_capture_png() {
            const v = document.getElementById('letara_vid');
            const c = document.getElementById('letara_can');
            const ctx = c.getContext('2d');
            c.width = v.videoWidth || 640;
            c.height = v.videoHeight || 480;
            ctx.drawImage(v, 0, 0, c.width, c.height);
            return c.toDataURL('image/png');
        }
        window.addEventListener('beforeunload', () => {
            const v = document.getElementById('letara_vid');
            if (v && v.srcObject) v.srcObject.getTracks().forEach(t => t.stop());
        });
        </script>
    ''')
