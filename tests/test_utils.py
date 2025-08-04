# Tests for utility functions.
import numpy as np
from rashomon_emotion import utils
import os
import tempfile

def test_downsample_signal():
    sig = np.random.randn(4, 512)  # 4 channels, 512 samples
    ds = utils.downsample_signal(sig, target_len=256)
    assert ds.shape == (4, 256)

def test_pickle_io():
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.close()

    data = {"a": 1, "b": [1, 2, 3]}
    utils.save_pickle(data, tmp_file.name)
    loaded = utils.load_pickle(tmp_file.name)
    assert loaded == data

    os.remove(tmp_file.name)
