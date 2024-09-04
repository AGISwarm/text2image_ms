"""
This module is the entry point for the text2image_ms service.

"""

import asyncio
import base64
import logging
import traceback
from io import BytesIO
from pathlib import Path

import hydra
import uvicorn
from AGISwarm.asyncio_queue_manager import AsyncIOQueueManager, RequestStatus
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic.main import BaseModel

from .diffusion_pipeline import Text2ImagePipeline
from .typing import Text2ImageGenerationConfig, Text2ImagePipelineConfig


class Text2ImageApp:
    """
    A class to represent the Text2Image service.
    """

    def __init__(self, config: Text2ImagePipelineConfig):
        self.app = FastAPI()
        self.setup_routes()
        self.queue_manager = AsyncIOQueueManager()
        self.text2image_pipeline = Text2ImagePipeline(config)

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

        self.ws_router = APIRouter()
        self.ws_router.add_websocket_route("/ws", self.generate)
        self.ws_router.post("/abort")(self.abort)
        self.app.include_router(self.ws_router)

    async def generate(self, websocket: WebSocket):
        """
        Generate an image from a text prompt using the Text2Image pipeline.
        """

        await websocket.accept()

        try:
            while True:
                await asyncio.sleep(0.01)
                data = await websocket.receive_text()
                print(data)
                gen_config = Text2ImageGenerationConfig.model_validate_json(data)
                generator = self.queue_manager.queued_generator(
                    self.text2image_pipeline.generate
                )
                request_id = generator.request_id
                interrupt_event = self.queue_manager.abort_map[request_id]

                async for step_info in generator(
                    gen_config, interrupt_event=interrupt_event
                ):
                    await asyncio.sleep(0.01)
                    print(step_info)
                    if step_info["status"] == RequestStatus.WAITING:
                        await websocket.send_json(step_info)
                        continue
                    if step_info["status"] != RequestStatus.RUNNING:
                        await websocket.send_json(step_info)
                        break
                    latents = step_info["image"]
                    image_io = BytesIO()
                    latents.save(image_io, "PNG")
                    dataurl = "data:image/png;base64," + base64.b64encode(
                        image_io.getvalue()
                    ).decode("ascii")
                    await websocket.send_json(
                        {
                            "request_id": request_id,
                            "status": RequestStatus.RUNNING,
                            "step": step_info["step"],
                            "total_steps": step_info["total_steps"],
                            "latents": dataurl,
                            "shape": latents.size,
                        }
                    )
        except Exception as e:  # pylint: disable=broad-except
            logging.error(e)
            traceback.print_exc()
            await websocket.send_json(
                {
                    "status": RequestStatus.ERROR,
                    "message": str(e),  ### loggging
                }
            )
        finally:
            await websocket.close()

    class AbortRequest(BaseModel):
        """Abort request"""

        request_id: str

    async def abort(self, request: AbortRequest):
        """Abort generation"""
        print(f"Aborting request {request.request_id}")
        await self.queue_manager.abort_task(request.request_id)

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
    uvicorn.run(text2image_app.app, host="127.0.0.1", port=8002, log_level="debug")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
