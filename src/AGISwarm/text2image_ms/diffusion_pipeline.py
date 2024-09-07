"""
This module contains the Stable Diffusion Pipeline class 
that is used to generate images from text prompts using the Stable Diffusion model.
"""

from typing import Callable, Optional

import torch
from diffusers import StableDiffusionPipeline

from .typing import DiffusionConfig, Text2ImageGenerationConfig


# pylint: disable=too-few-public-methods
class Text2ImagePipeline:
    """
    A class to generate images from text prompts using the Stable Diffusion model.

    Args:
        config (Text2ImagePipelineConfig):
            The configuration for the Diffusion Pipeline initialization.
                - model (str): The model to use for generating the image.
                - dtype (str): The data type to use for the model.
                - device (str): The device to run the model on.
                - safety_checker (str | None): The safety checker to use for the model.
                - requires_safety_checker (bool): Whether the model requires a safety checker.
                - low_cpu_mem_usage (bool): Whether to use low CPU memory usage.
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            config.model,
            torch_dtype=getattr(torch, config.dtype),
            safety_checker=config.safety_checker,
            requires_safety_checker=config.requires_safety_checker,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
        ).to(config.device)

        # sfast_config = CompilationConfig.Default()
        # sfast_config.enable_cuda_graph = True
        self.pipeline.vae.enable_tiling()
        self.pipeline.vae.enable_slicing()
        # self.pipeline.enable_sequential_cpu_offload()

        # Warmup the model
        # self.pipeline(
        #     prompt="warmup",
        #     negative_prompt="warmup",
        #     num_inference_steps=40,
        # )

    def generate(
        self,
        gen_config: Text2ImageGenerationConfig,
        callback_on_step_end: Optional[Callable[[dict], None]] = None,
    ):
        """
        Generate an image from a text prompt using the Text2Image pipeline.

        Args:
            gen_config (Text2ImageGenerationConfig):
                The configuration for the Text2Image Pipeline generation.
                    - prompt (str): The text prompt to generate the image from.
                    - negative_prompt (str): The negative text prompt to generate the image from.
                    - num_inference_steps (int): The number of inference steps to run.
                    - guidance_scale (float): The guidance scale to use for the model.
                    - seed (int): The seed to use for the model.
                    - width (int): The width of the image to generate.
                    - height (int): The height of the image to generate.
            callback_on_step_end (Optional[Callable[[dict], None]):
                The callback function to call on each step end.
        """
        return {
            "image": self.pipeline(
                prompt=gen_config.prompt,
                negative_prompt=gen_config.negative_prompt,
                num_inference_steps=gen_config.num_inference_steps,
                guidance_scale=gen_config.guidance_scale,
                generator=torch.Generator().manual_seed(gen_config.seed),
                width=gen_config.width,
                height=gen_config.height,
                callback_on_step_end=callback_on_step_end,
            )["images"][0]
        }
