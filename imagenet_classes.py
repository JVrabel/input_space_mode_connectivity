# ImageNet 2012 class mapping (numeric label 0-999 <-> WordNet ID)
# Loads from parent repo data/imagenet_classes.py if available.
# Otherwise copy data/imagenet_classes.py here or pass --imagenet_classes path.

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PARENT_DATA = _SCRIPT_DIR.parent / "data" / "imagenet_classes.py"

if _PARENT_DATA.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("imagenet_full", _PARENT_DATA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    IMAGENET2012_CLASSES = mod.IMAGENET2012_CLASSES
else:
    raise ImportError(
        f"ImageNet classes not found. Copy data/imagenet_classes.py to this package, "
        f"or ensure parent repo has data/imagenet_classes.py. Expected: {_PARENT_DATA}"
    )

CLASS_TO_IDX = {k: i for i, k in enumerate(IMAGENET2012_CLASSES.keys())}
