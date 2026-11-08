"""Optional CPU ray kernels. Python remains the reference/fallback backend."""
try:
    from robot_mapping import _ray_native as kernels
except ImportError:
    kernels = None
