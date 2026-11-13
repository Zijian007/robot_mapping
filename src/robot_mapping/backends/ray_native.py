"""Optional CPU ray kernels. Python remains the reference/fallback backend."""

try:
    from robot_mapping import _ray_native as kernels
except ImportError:
    kernels = None
else:
    if getattr(kernels, "API_VERSION", 0) < 10:
        raise RuntimeError("robot_mapping._ray_native requires API >= 10; rebuild robot-mapping")
