"""FLUXoFORUM: frame-streaming Deforum animation for FLUX.2 Klein."""

from .config import GenerationConfig
from .engine import AnimationEngine
from .jobs import JobStore

__version__ = "1.0.0"
__all__ = ["AnimationEngine", "GenerationConfig", "JobStore", "__version__"]

