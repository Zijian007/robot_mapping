"""Optional CPU ray kernels. Python remains the reference/fallback backend."""

try:
    from robot_mapping import _ray_native as kernels
except ImportError:
    kernels = None
else:
    if getattr(kernels, "API_VERSION", 0) < 8:
        raise RuntimeError("robot_mapping._ray_native requires API >= 8; rebuild robot-mapping")
