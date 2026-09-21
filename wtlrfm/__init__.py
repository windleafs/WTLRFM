"""WTLR network modules adapted from /data/zhuangyang/WTLRFM (verbatim copies).

Source files (copied unchanged, only this __init__ is new):
    wtlr_encoder.py            <- WTLRFM/models/wtlr_encoder.py
    liif_encoder.py            <- WTLRFM/models/liif_encoder.py
    sr3_modules/wtlr_unet.py   <- WTLRFM/models/sr3_modules/wtlr_unet.py
    sr3_modules/speckle_layer.py, kan_layers.py  (imported by wtlr_unet.py)
"""
from .wtlr_encoder import WTLREncoder
from .sr3_modules.wtlr_unet import WTLRUNet

__all__ = ["WTLREncoder", "WTLRUNet"]
