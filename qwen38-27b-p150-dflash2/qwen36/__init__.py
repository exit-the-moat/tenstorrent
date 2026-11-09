# ep2: QWEN36_PRESET=best2card|best1card expands to the gated switch set before any module reads its switches
from models.demos.blackhole.qwen36.presets import apply as _apply_preset

_apply_preset()
