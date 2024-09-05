"""
This module is the entry point for the text2image_ms service.

"""

import asyncio
import base64
import logging
import multiprocessing as mp
import traceback
from functools import partial
from io import BytesIO
from pathlib import Path

import hydra
import nest_asyncio
import numpy as np
import torch
import uvicorn
from AGISwarm.asyncio_queue_manager import AsyncIOQueueManager, RequestStatus
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic.main import BaseModel

from .diffusion_pipeline import Text2ImagePipeline
from .typing import Config, DiffusionConfig, GUIConfig, Text2ImageGenerationConfig


def _to_task(future: asyncio.Future, as_task: bool, loop: asyncio.AbstractEventLoop):
    if not as_task or isinstance(future, asyncio.Task):
        return future
    return loop.create_task(future)


def asyncio_run(future, as_task=True):
    """
    A better implementation of `asyncio.run`.

    :param future: A future or task or call of an async method.
    :param as_task: Forces the future to be scheduled as task (needed for e.g. aiohttp).
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(_to_task(future, as_task, loop))
    else:
        nest_asyncio.apply(loop)
        return asyncio.run(_to_task(future, as_task, loop))


class Text2ImageApp:
    """
    A class to represent the Text2Image service.
    """

    def __init__(self, config: DiffusionConfig, gui_config: GUIConfig):
        self.app = FastAPI()
        self.setup_routes()
        self.queue_manager = AsyncIOQueueManager()
        self.text2image_pipeline = Text2ImagePipeline(config)
        self.latent_update_frequency = gui_config.latent_update_frequency

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

    @staticmethod
    def send_image(websocket: WebSocket, image: Image.Image, **kwargs):
        """
        Send an image to the client.
        """
        image_io = BytesIO()
        image.save(image_io, "PNG")
        dataurl = "data:image/png;base64," + base64.b64encode(
            image_io.getvalue()
        ).decode("ascii")
        asyncio_run(
            websocket.send_json(
                {
                    "image": dataurl,
                    "shape": image.size,
                }
                | kwargs
            )
        )

    @staticmethod
    def diffusion_pipeline_step_callback(
        websocket: WebSocket,
        request_id: str,
        abort_event: asyncio.Event,
        total_steps: int,
        latent_update_frequency: int,
        pipeline,
        step: int,
        timestep: int,
        callback_kwargs: dict,
    ):
        """Callback для StableDiffusionPipeline"""
        if abort_event.is_set():
            raise asyncio.CancelledError("Diffusion pipeline aborted")
        if step == 0 or step != total_steps and step % latent_update_frequency != 0:
            return {"latents": callback_kwargs["latents"]}
        with torch.no_grad():
            image = pipeline.decode_latents(callback_kwargs["latents"].clone())[0]
        image = Image.fromarray((image * 255).astype(np.uint8))
        Text2ImageApp.send_image(
            websocket,
            image,
            request_id=request_id,
            status=RequestStatus.RUNNING,
            step=step,
            total_steps=total_steps,
        )
        return {"latents": callback_kwargs["latents"]}

    async def generate(self, websocket: WebSocket):
        """
        Generate an image from a text prompt using the Text2Image pipeline.
        """

        await websocket.accept()

        try:
            while True:
                await asyncio.sleep(0.01)
                data = await websocket.receive_text()
                # Read generation config
                gen_config = Text2ImageGenerationConfig.model_validate_json(data)
                # Enqueue the task (without starting it)
                queued_task = self.queue_manager.queued_task(
                    self.text2image_pipeline.generate
                )

                # request_id and interrupt_event are created by the queued_generator
                request_id = queued_task.request_id
                abort_event = self.queue_manager.abort_map[request_id]

                # Diffusion step callback
                callback_on_step_end = partial(
                    self.diffusion_pipeline_step_callback,
                    websocket,
                    request_id,
                    abort_event,
                    gen_config.num_inference_steps,
                    self.latent_update_frequency,
                )

                # Start the generation task
                try:
                    async for step_info in queued_task(
                        gen_config, callback_on_step_end
                    ):
                        if "status" not in step_info:  # Task's return value.
                            Text2ImageApp.send_image(
                                websocket,
                                step_info["image"],
                                request_id=request_id,
                                status=RequestStatus.FINISHED,
                            )
                            break
                        if (
                            step_info["status"] == RequestStatus.WAITING
                        ):  # Queuing info returned
                            await websocket.send_json(step_info)
                            continue
                        if (
                            step_info["status"] != RequestStatus.RUNNING
                        ):  # Queuing info returned
                            await websocket.send_json(step_info)
                            break
                except asyncio.CancelledError as e:
                    logging.info(e)
                    await websocket.send_json(
                        {
                            "status": RequestStatus.ABORTED,
                            "request_id": request_id,
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

    async def gui(self):
        """
        Get the GUI for the Text2Image service.
        """
        print("GUI")
        path = Path(__file__).parent / "app" / "gui.html"
        return FileResponse(path)


@hydra.main(config_name="config")
def main(config: Config):
    """
    The main function for the Text2Image service.
    """
    text2image_app = Text2ImageApp(config.diffusion_config, config.gui_config)
    uvicorn.run(
        text2image_app.app,
        host="127.0.0.1",
        port=8002,
        log_level="debug",
        loop="asyncio",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
