"""
This module is the entry point for the text2image_ms service.

"""

import asyncio
import base64
import logging
import traceback
import uuid
from io import BytesIO
from pathlib import Path

import hydra
import uvicorn
from fastapi import Body, FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from hydra.core.config_store import ConfigStore
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from .diffusion_pipeline import Text2ImagePipeline
from .typing import Text2ImageGenerationConfig, Text2ImagePipelineConfig


class Text2ImageApp:
    """
    A class to represent the Text2Image service.
    """

    def __init__(self, config: Text2ImagePipelineConfig):
        self.app = FastAPI(debug=True)
        self.text2image_pipeline = Text2ImagePipeline(config)
        self.setup_routes()

    def setup_routes(self):
        """
        Set up the routes for the Text2Image service.
        """
        self.app.get("/", response_class=HTMLResponse)(self.gui)
        self.app.mount(
            "/static",
            StaticFiles(directory=Path(__file__).parent / "app", html=True),
            name="static",
        )
        self.app.add_websocket_route("/ws", self.generate)

    async def generate(self, websocket: WebSocket):
        """
        Generate an image from a text prompt using the Text2Image pipeline.
        """

        await websocket.accept()
        
        try:
            while True:
                await asyncio.sleep(0.1)
                data = await websocket.receive_text()
                print (data)
                gen_config = Text2ImageGenerationConfig.model_validate_json(data)
                request_id = str(uuid.uuid4())
                async for step_info in self.text2image_pipeline.generate(request_id, gen_config):
                    if step_info['type'] == 'waiting':
                        await websocket.send_json(step_info)
                        continue
                    latents = step_info['image']
                    
                    # Конвертируем латенты в base64
                    image_io = BytesIO()
                    latents.save(image_io, 'PNG')
                    dataurl = 'data:image/png;base64,' + base64.b64encode(image_io.getvalue()).decode('ascii')
                    # Отправляем инфу о прогрессе и латенты
                    await websocket.send_json({
                        "type": "generation_step",
                        "step": step_info['step'],
                        "total_steps": step_info['total_steps'],
                        "latents": dataurl,
                        "shape": latents.size
                    })
                
                await websocket.send_json({
                    "type": "generation_complete"
                })
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        finally:
            await websocket.close()

    def gui(self):
        """
        Get the GUI for the Text2Image service.
        """
        print("GUI")
        path = Path(__file__).parent / "app" / "gui.html"
        return FileResponse(path)


@hydra.main(config_name="config")
def main(config: Text2ImagePipelineConfig):
    """
    The main function for the Text2Image service.
    """
    text2image_app = Text2ImageApp(config)
    uvicorn.run(text2image_app.app, host="localhost", port=8002, log_level="debug")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
