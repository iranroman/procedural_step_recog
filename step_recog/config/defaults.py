#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

_C = CfgNode()

# -----------------------------------------------------------------------------
# TRAIN options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 0.001
_C.TRAIN.EPOCHS = 5

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.HIDDEN_SIZE = 1024
_C.MODEL.CONTEXT_LENGTH = 4
_C.MODEL.OUTPUT_DIM = 33

# -----------------------------------------------------------------------------
# Dataset options
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.NAME = ''
_C.DATASET.LOCATION = ''
_C.DATASET.VIDEO_LAYER = ''
_C.DATASET.TR_ANNOTATIONS_FILE = ''
_C.DATASET.VL_ANNOTATIONS_FILE = ''
_C.DATASET.HOP_SIZE = 0.5
_C.DATASET.FPS = 30

# -----------------------------------------------------------------------------
# Dataloader options
# -----------------------------------------------------------------------------
_C.DATALOADER = CfgNode()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.PIN_MEMORY = True


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C
