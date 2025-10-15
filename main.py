from nicegui import ui
import pages  # importing registers all routes

# Windows-safe guard (covers multiprocessing import name)
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='LeTARA', reload=False)
