# -------------------------------------------------------------------------
# Copyright (c) SpacemiT Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""
Ensure that dependencies are available and then load the extension module.
"""

from . import _ld_preload  # noqa: F401

from .bianbuai_pybind11_state import *  # noqa
