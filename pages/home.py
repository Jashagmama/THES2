from nicegui import ui
from components.layout import layout
from pathlib import Path

@ui.page('/')
def home_page():
    layout('home')
    ui.label('Welcome to LeTARA!').classes('text-2xl font-bold mt-4 text-[#22c55e]')

    with ui.row().classes('mt-4 gap-4 flex-wrap'):
        
        with ui.card().classes('p-4 bg-[#166534]/70 text-white'):
            ui.icon('home').classes('text-3xl text-[#4ade80]')
            ui.label('Quick Actions').classes('font-semibold')
            ui.button('Go to Profile', on_click=lambda: ui.navigate.to('/profile')).props('color=positive')
            ui.button('Open Camera', on_click=lambda: ui.navigate.to('/camera')).props('color=secondary')

        with ui.card().classes('p-4 bg-[#166534]/70 text-white w-[360px] max-w-full'):
            ui.label('Upload & Results').classes('font-semibold')
            ui.button('Go to Upload Page', on_click=lambda: ui.navigate.to('/upload')).props('color=positive').classes('mb-2')

            uploads = sorted(Path('./data/uploads').glob('*.png'), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            if uploads:
                ui.label('Recent uploads:').classes('text-sm text-gray-200')
                for p in uploads:
                    ui.label(f'• {p.name}').classes('text-sm text-gray-200')
            else:
                ui.label('No uploads yet.').classes('text-sm text-gray-300')
