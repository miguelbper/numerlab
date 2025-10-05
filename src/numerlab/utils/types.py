from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

Path_ = str | Path
Metrics = dict[str, float]

# torch: X, y, m, e
Batch = tuple[
    Tensor,  # features
    Tensor,  # target
    Tensor,  # meta
    Tensor,  # era
]

# numpy: X, y, m, e
Data = tuple[
    NDArray[np.int8],  # features
    NDArray[np.float32],  # target
    NDArray[np.float32],  # meta
    NDArray[np.int32],  # era
]
