from __future__ import annotations

if __name__ == "__main__":
    from endometrial_app.streamlit_ui import main

    main()
else:
    import gradio as gr

    from endometrial_app.api import create_api_app
    from endometrial_app.service import PredictionService
    from endometrial_app.ui import build_ui

    service = PredictionService.from_settings()
    api_app = create_api_app(service)
    ui = build_ui(service)
    app = gr.mount_gradio_app(api_app, ui, path="/")
