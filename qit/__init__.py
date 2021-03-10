"""qp is a library for manaing and converting between different representations of distributions"""

import os

try:
    from .version import get_git_version
    __version__ = get_git_version()
except Exception as message: #pragma: no cover
    print(message)

from .axis import Axis
from .estimator import Estimator
from .posterior import Posterior
from .binned_like import BinnedLike, Likelihood

from . import like_funcs
from . import plotting
