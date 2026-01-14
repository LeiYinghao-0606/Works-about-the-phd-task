# -*- coding: utf-8 -*-
"""
Reproducibility utilities.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed python / numpy / torch.
    If deterministic=True, set cuDNN flags for determinism (may slow down).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

