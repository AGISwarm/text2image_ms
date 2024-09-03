"""
This module contains the Stable Diffusion Pipeline class 
that is used to generate images from text prompts using the Stable Diffusion model.
"""

import asyncio
import threading
from functools import partial
from typing import Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.callbacks import PipelineCallback
from PIL import Image

from .typing import Text2ImageGenerationConfig, Text2ImagePipelineConfig
from .utils import generation_request_queued_func


class DiffusionIteratorStreamer:
    
    def __init__(self, timeout: Optional[Union[float, int]] = None):
        self.latents_stack = []
        self.stop_signal: Optional[str] = None
        self.timeout = timeout
        self.current_step = 0
        self.total_steps = 0
        self.stop = False

    def put(self, latents: torch.Tensor):
        """Метод для добавления латентов в очередь"""
        self.latents_stack.append(latents.cpu().numpy())

    def end(self):
        """Метод для сигнализации окончания генерации"""
        self.stop = True

    def __aiter__(self) -> "DiffusionIteratorStreamer":
        return self

    async def __anext__(self) -> torch.Tensor | str:
        while len(self.latents_stack) == 0:
            await asyncio.sleep(0.1)
        latents = self.latents_stack.pop()
        if self.stop:
            raise StopAsyncIteration()
        return latents

    def callback(
        self,
        pipeline: DiffusionPipeline,
        step: int,
        timestep: int,
        callback_kwargs: dict,
    ):
        """Callback для StableDiffusionPipeline"""
        self.current_step = step
        self.put(callback_kwargs["latents"])
        return {"latents": callback_kwargs["latents"]}

    def stream(
        self,
        pipe: StableDiffusionPipeline,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        width: int,
        height: int,
    ):
        """Method to stream the diffusion pipeline"""
        self.total_steps = num_inference_steps

        def run_pipeline():
            pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                width=width,
                height=height,
                callback_on_step_end=self.callback,  # type: ignore
            )
            self.end()

        thread = threading.Thread(target=run_pipeline)
        thread.start()
        return self


class Text2ImagePipeline:
    """
    A class to generate images from text prompts using the Stable Diffusion model.

    Args:
        config (Text2ImagePipelineConfig): The configuration for the Diffusion Pipeline initialization.
        - model (str): The model to use for generating the image.
        - dtype (str): The data type to use for the model.
        - device (str): The device to run the model on.
        - safety_checker (str | None): The safety checker to use for the model.
        - requires_safety_checker (bool): Whether the model requires a safety checker.
        - low_cpu_mem_usage (bool): Whether to use low CPU memory usage.
    """

    def __init__(self, config: Text2ImagePipelineConfig):
        self.config = config
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            config.model,
            torch_dtype=getattr(torch, config.dtype),
            safety_checker=config.safety_checker,
            requires_safety_checker=config.requires_safety_checker,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
        ).to(config.device)

        self.pipeline.vae.enable_tiling()
        self.pipeline.vae.enable_slicing()
        # self.pipeline.enable_sequential_cpu_offload()

    @partial(generation_request_queued_func, wait_time=0.1)
    async def generate(
        self,
        request_id: str,
        gen_config: Text2ImageGenerationConfig
    ):
        """
        Generate an image from a text prompt using the Text2Image pipeline.

        Args:
            gen_config (Text2ImageGenerationConfig): The configuration for the Text2Image Pipeline generation.
            - prompt (str): The text prompt to generate the image from.
            - negative_prompt (str): The negative text prompt to generate the image from.
            - num_inference_steps (int): The number of inference steps to run.
            - guidance_scale (float): The guidance scale to use for the model.
            - seed (int): The seed to use for the model.
            - width (int): The width of the image to generate.
            - height (int): The height of the image to generate.

        Yields:
            dict: A dictionary containing the step information for the generation.
            - step (int): The current step of the generation.
            - total_steps (int): The total number of steps for the generation.
            - image (PIL.Image): The generated image
        """
        streamer = DiffusionIteratorStreamer()
        streamer.stream(
            self.pipeline,
            prompt=gen_config.prompt,
            negative_prompt=gen_config.negative_prompt,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            seed=gen_config.seed,
            width=gen_config.width,
            height=gen_config.height,
        )

        async for latents in streamer:
            if latents is None:
                await asyncio.sleep(0.1)
                continue

            latents = torch.tensor(latents, device=self.config.device)
            with torch.no_grad():
                image = self.pipeline.decode_latents(latents)[0]
            image = Image.fromarray((image * 255).astype(np.uint8))
            await asyncio.sleep(0.1)
            yield {
                "type": "generation_step",
                "request_id": request_id,
                "step": streamer.current_step,
                "total_steps": streamer.total_steps,
                "image": image,
            }
