# model_utils/cosre/__init__.py
import torch.nn as nn
from .config import CoSReConfig
from .cosine_codec import CosineCodecConfig, BlockwiseCosineCodec
from .shared_residual import SharedResidualConfig, SharedResidualFactorizer
from .controller import CoSReController, CoSReControllerConfig
from .hooks import CoSReGenerationHook, CoSReHookConfig, attach_cosre_generate_to_model

__all__ = [
    "CoSReConfig",
    "CosineCodecConfig",
    "BlockwiseCosineCodec",
    "SharedResidualConfig",
    "SharedResidualFactorizer",
    "CoSReControllerConfig",
    "CoSReController",
    "CoSReHookConfig",
    "CoSReGenerationHook",
    "attach_cosre_generate_to_model",
]


def attach_cosre(model: nn.Module, enable: bool, image_hw=(32, 32), image_mask_fn=None):
    hook = CoSReGenerationHook.attach(
        model=model,
        enable=enable,
        image_hw=image_hw,
        image_mask_fn=image_mask_fn,
        strict_image_mask=True,
        force_use_cache=True,
    )
    attach_cosre_generate_to_model(model, hook, method_name="cosre_generate")
    return model
