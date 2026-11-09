"""Optional CPU ray kernels. Python remains the reference/fallback backend."""
try:
    from robot_mapping import _ray_native as kernels
except ImportError:
    kernels = None
else:
    # An older in-place extension must not leave new Python code using a
    # partially compatible backend. Managed runtime verification rejects fallback.
    if getattr(kernels, "API_VERSION", 0) < 5:
        kernels = None
