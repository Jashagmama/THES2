from nicegui import ui

def layout(active: str):
    with ui.header().classes('items-center justify-between px-4 bg-[#14532d] text-white'):
        ui.label('🌿 LeTARA Dashboard').classes('text-lg font-semibold')
        with ui.row().classes('gap-4'):
            def nav_link(title, href, key):
                cls = 'text-[#86efac] font-medium underline' if active == key else 'text-gray-200 hover:text-green-200'
                ui.link(title, href).classes(cls)
            nav_link('Home', '/', 'home')
            nav_link('Profile', '/profile', 'profile')
            nav_link('Camera', '/camera', 'camera')
            nav_link('Upload', '/upload', 'upload')     

    with ui.footer().classes('justify-center text-gray-300 bg-[#14532d]'):
        ui.label('© 2025 - LeTARA')
