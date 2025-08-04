"""
Rashomon Emotion Recognition Package
------------------------------------
A modular deep learning framework for EEG-based emotion recognition
using the Rashomon effect.
"""

from . import utils
from . import preprocessing
from . import features
from . import graph_utils
from . import model
from . import interpretability
from . import rashomon_runner
from . import config

__all__ = [
    "utils",
    "preprocessing",
    "features",
    "graph_utils",
    "model",
    "interpretability",
    "rashomon_runner",
    "config"
]
