"""
This module is the entry point for the text2image_ms service.

"""

import os
from pathlib import Path

import hydra
import uvicorn

from .app import Text2ImageApp
from .typing import Config


@hydra.main(
    version_base=None,
    config_name="config",
    config_path=str(Path(os.getcwd()) / "config"),
)
def main(config: Config):
    """
    The main function for the Text2Image service.
    """
    text2image_app = Text2ImageApp(config.t2i_model_config, config.gui_config)
    uvicorn.run(
        text2image_app.app,
        host="127.0.0.1",
        port=8002,
        log_level="debug",
        loop="asyncio",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
