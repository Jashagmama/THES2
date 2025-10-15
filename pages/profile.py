from nicegui import ui
from components.layout import layout

@ui.page('/profile')
def profile_page():
    layout('profile')
    ui.label('Profile').classes('text-2xl font-bold mt-4 text-[#22c55e]')

    with ui.card().classes('mt-4 p-4 w-full max-w-xl bg-[#166534]/70 text-white'):
        name = ui.input('Full name', placeholder='e.g., Audrey').classes('w-full text-white')
        email = ui.input('Email', placeholder='name@example.com').classes('w-full text-white')
        ui.textarea('About you').props('rows=4').classes('w-full text-white')

        def save_profile():
            ui.notify(f'Saved ✓  Name: {name.value}, Email: {email.value}', color='positive')

        ui.button('Save', on_click=save_profile).props('color=positive').classes('mt-2')
