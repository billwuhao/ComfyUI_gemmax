from pathlib import Path

import uvicorn
from faicons import icon_svg as icon
from fire import Fire
from shiny import App, reactive, render, ui

from quickmt import Translator
from quickmt.hub import hf_download, hf_list

t = None


def runapp(port: int = 8000, host: str = "127.0.0.1"):
    ui.navbar_options(
        bg="red",
    )
    app_ui = ui.page_navbar(
        ui.nav_panel(
            None,
            ui.layout_columns(
                ui.card(
                    ui.h4("Input Text"),
                    ui.input_text_area(
                        "input_text",
                        "",
                        value="",
                        width="100%",
                        height="600px",
                    ),
                    ui.input_action_button(
                        "translate_button", "Translate!", class_="btn-primary"
                    ),
                ),
                ui.card(ui.h4("Translation"), ui.output_ui("translate")),
            ),
        ),
        ui.nav_spacer(),
        ui.nav_control(
            ui.input_dark_mode(
                id="darkmode_toggle", mode="dark", style="padding-top: 10px;"
            ),
        ),
        ui.nav_control(
            ui.a(
                icon("github", height="30px", width="30px", fill="#17a2b8"),
                href="https://github.com/quickmt/quickmt",
                target="_blank",
                class_="btn btn-link",
            ),
        ),
        sidebar=ui.sidebar(
            ui.tooltip(
                ui.input_selectize(
                    "model",
                    "Select model",
                    choices=[i.split("/")[1] for i in hf_list()],
                ),
                "QuickMT model to use. quickmt-fr-en will translate from French (fr) to English (en)",
            ),
            ui.tooltip(
                ui.input_text(
                    "model_folder", "Model Folder", value=str(Path(".").absolute())
                ),
                "Folder where QuickMT models are (or will be) stored.",
            ),
            ui.tooltip(
                ui.input_slider(
                    "beam_size", "Beam size", min=1, max=8, step=1, value=2
                ),
                "Balances speed and quality. 1 for fastest speed, 8 for highest quality, in between for a balance.",
            ),
            ui.tooltip(
                ui.input_numeric(
                    "num_threads", "CPU Threads", min=1, max=16, step=1, value=4
                ),
                "Number of CPU threads to use for translation. Does not affect speed when using GPU.",
            ),
            ui.tooltip(
                ui.input_selectize(
                    "compute_device",
                    "Compute Device",
                    choices=["auto", "cpu", "cuda"],
                    selected="cpu",
                ),
                "Auto will use the GPU if available, otherwise will use CPU.",
            ),
            width="350px",
        ),
        title=ui.h2("QuickMT"),
        window_title="QuickMT",
        theme=ui.Theme.from_brand(__file__),
        navbar_options=ui.navbar_options(underline=False, theme="auto"),
    )

    def server(input, output, session):
        @render.ui
        @reactive.event(input.quickmt_model_download)  # Take a dependency on the button
        def model_download_output():
            print(f"Downloading {input.model()} to {input.model_folder()}")
            hf_download(
                model_name="quickmt/" + input.model(),
                output_dir=Path(input.model_folder()) / input.model(),
            )
            return "Model downloaded"

        @render.ui
        @reactive.event(input.translate_button)  # Take a dependency on the button
        def translate():
            global t
            model_path = Path(input.model_folder()) / input.model()
            try:
                if t is None or str(input.model()) != str(Path(t.model_path).name):
                    print(f"Loading model {input.model()}")
                    t = Translator(
                        str(model_path),
                        device=input.compute_device(),
                        inter_threads=int(input.num_threads()),
                    )
                if len(input.input_text()) == 0:
                    return ""

                return [
                    ui.p(i)
                    for i in t(
                        input.input_text().splitlines(), beam_size=input.beam_size()
                    )
                ]

            except:
                return [
                    ui.value_box(
                        title=f"Ensure model is downloaded to {input.model_folder()}",
                        value="Failed to load model",
                        showcase=icon("bug"),
                    ),
                    ui.input_action_button(
                        "quickmt_model_download", "Download Model", class_="btn-primary"
                    ),
                    ui.output_ui("model_download_output"),
                ]

    app = App(app_ui, server)
    uvicorn.run(app, port=port, host=host)


def main():
    Fire(runapp)


if __name__ == "__main__":
    main()
