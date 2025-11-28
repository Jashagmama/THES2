from nicegui import ui
import sys

# Import pages AFTER nicegui but before ui.run()
# This ensures ui is available when @ui.page decorators execute
import pages  # importing registers all routes

# def cleanup():
#     """Cleanup resources on exit"""
#     print("\n🧹 Cleaning up...")
#     try:
#         import tensorflow as tf
#         tf.keras.backend.clear_session()
#     except:
#         pass

# Windows-safe guard (covers multiprocessing import name)
if __name__ in {"__main__", "__mp_main__"}:
    try:
        ui.run(
            title='LeTARA',
            reload=False,
            ssl_certfile='localhost.crt',
            ssl_keyfile='localhost.key',
            port=8000
        )
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
        # cleanup()
        sys.exit(0)
