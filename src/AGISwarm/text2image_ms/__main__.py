"""
This module is the entry point for the text2image_ms service.

"""

import os
from pathlib import Path

import hydra
import uvicorn

from .app import Text2ImageApp
from .typing import Text2ImageConfig


@hydra.main(
    version_base=None,
    config_name="config",
    config_path=str(Path(os.getcwd()) / "config"),
)
def main(config: Text2ImageConfig):
    """
    The main function for the Text2Image service.
    """
    text2image_app = Text2ImageApp(
        config.hf_model_name, config.t2i_model_config, config.gui_config
    )
    uvicorn.run(
        text2image_app.app,
        host=config.uvicorn_config.host,
        port=config.uvicorn_config.port,
        log_level=config.uvicorn_config.log_level,
        loop=config.uvicorn_config.loop,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
