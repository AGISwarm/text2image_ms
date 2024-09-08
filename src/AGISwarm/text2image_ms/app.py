"""
This module is the entry point for the text2image_ms service.

"""

import asyncio
import base64
import logging
from functools import partial
from io import BytesIO
from pathlib import Path

from AGISwarm.asyncio_queue_manager import AsyncIOQueueManager, TaskStatus
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic.main import BaseModel

from .diffusion_pipeline import Text2ImagePipeline
from .typing import DiffusionConfig, GUIConfig, Text2ImageGenerationConfig
from .utils import asyncio_run


class Text2ImageApp:
    """
    A class to represent the Text2Image service.
    """

    def __init__(
        self, hf_model_name: str, config: DiffusionConfig, gui_config: GUIConfig
    ):
        self.app = FastAPI()
        self.setup_routes()
        self.queue_manager = AsyncIOQueueManager(sleep_time=0.0001)
        self.text2image_pipeline = Text2ImagePipeline(hf_model_name, config)
        self.latent_update_frequency = gui_config.latent_update_frequency
        self.start_abort_lock = asyncio.Lock()

    def setup_routes(self):
        """
        Set up the routes for the Text2Image service.
        """
        self.app.get("/", response_class=HTMLResponse)(self.gui)
        self.app.mount(
            "/static",
            StaticFiles(directory=Path(__file__).parent / "gui", html=True),
            name="static",
        )

        self.ws_router = APIRouter()
        self.ws_router.add_websocket_route("/ws", self.generate)
        self.app.post("/abort")(self.abort)
        self.app.include_router(self.ws_router)

    @staticmethod
    def pil2dataurl(image: Image.Image):
        """
        Convert a PIL image to a data URL.
        """
        image_io = BytesIO()
        image.save(image_io, "PNG")
        dataurl = "data:image/png;base64," + base64.b64encode(
            image_io.getvalue()
        ).decode("ascii")
        return dataurl

    @staticmethod
    def send_image(websocket: WebSocket, image: Image.Image, **kwargs):
        """
        Send an image to the client.
        """
        dataurl = Text2ImageApp.pil2dataurl(image)
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
    # pylint: disable=too-many-arguments
    def diffusion_pipeline_step_callback(
        websocket: WebSocket,
        task_id: str,
        abort_event: asyncio.Event,
        total_steps: int,
        latent_update_frequency: int,
        pipeline: Text2ImagePipeline,
        _diffusers_pipeline,
        step: int,
        _timestamp: int,
        callback_kwargs: dict,
    ):
        """Callback для StableDiffusionPipeline"""
        if abort_event.is_set():
            raise asyncio.CancelledError("Diffusion pipeline aborted")
        asyncio_run(asyncio.sleep(0.0001))
        if step == 0 or step != total_steps and step % latent_update_frequency != 0:
            return callback_kwargs
        image = pipeline.decode_latents(callback_kwargs["latents"])
        Text2ImageApp.send_image(
            websocket,
            image,
            task_id=task_id,
            status=TaskStatus.RUNNING,
            step=step,
            total_steps=total_steps,
        )
        return callback_kwargs

    async def generate(self, websocket: WebSocket):
        """
        Generate an image from a text prompt using the Text2Image pipeline.
        """

        await websocket.accept()

        try:
            while True:
                await asyncio.sleep(0.0001)
                data = await websocket.receive_text()
                async with self.start_abort_lock:
                    # Read generation config
                    gen_config = Text2ImageGenerationConfig.model_validate_json(data)
                    # Enqueue the task (without starting it)
                    queued_task = self.queue_manager.queued_task(
                        self.text2image_pipeline.generate
                    )
                    # task_id and interrupt_event are created by the queued_generator
                    task_id = queued_task.task_id
                    abort_event = self.queue_manager.abort_map[task_id]
                    await websocket.send_json(
                        {
                            "status": TaskStatus.STARTING,
                            "task_id": task_id,
                        }
                    )

                # Diffusion step callback
                callback_on_step_end = partial(
                    self.diffusion_pipeline_step_callback,
                    websocket,
                    task_id,
                    abort_event,
                    gen_config.num_inference_steps,
                    self.latent_update_frequency,
                    self.text2image_pipeline,
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
                                task_id=task_id,
                                status=TaskStatus.FINISHED,
                            )
                            break
                        if (
                            step_info["status"] == TaskStatus.WAITING
                        ):  # Queuing info returned
                            await websocket.send_json(step_info)
                            continue
                        if (
                            step_info["status"] != TaskStatus.RUNNING
                        ):  # Queuing info returned
                            await websocket.send_json(step_info)
                            break
                except asyncio.CancelledError as e:
                    logging.info(e)
                    await websocket.send_json(
                        {
                            "status": TaskStatus.ABORTED,
                            "task_id": task_id,
                        }
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logging.error(e)
                    await websocket.send_json(
                        {
                            "status": TaskStatus.ERROR,
                            "message": str(e),  ### loggging
                        }
                    )
        except Exception as e:  # pylint: disable=broad-except
            logging.error(e)
            await websocket.send_json(
                {
                    "status": TaskStatus.ERROR,
                    "message": str(e),  ### loggging
                }
            )
        finally:
            await websocket.close()

    class AbortRequest(BaseModel):
        """Abort request"""

        task_id: str

    async def abort(self, request: AbortRequest):
        """Abort generation"""
        print(f"ENTER ABORT Aborting request {request.task_id}")
        async with self.start_abort_lock:
            print(f"Aborting request {request.task_id}")
            await self.queue_manager.abort_task(request.task_id)

    async def gui(self):
        """
        Get the GUI for the Text2Image service.
        """
        print("GUI")
        path = Path(__file__).parent / "gui" / "gui.html"
        return FileResponse(path)
