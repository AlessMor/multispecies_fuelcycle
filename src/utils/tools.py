import numpy as np


def maybe_1d_float(values):
    """Return values as a 1D float array; scalars become length-1 arrays."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=float)
    return arr.reshape(-1).astype(float, copy=False)


def as_1d_float(values, name):
    """Coerce to 1D float array and require at least one element."""
    arr = maybe_1d_float(values)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def broadcast_1d(values, target_size, name):
    """Broadcast a 1D array to target size, allowing scalar expansion."""
    arr = as_1d_float(values, name)
    size = int(target_size)
    if size <= 0:
        raise ValueError("target_size must be positive")
    if arr.size == size:
        return arr
    if arr.size == 1:
        return np.full(size, float(arr[0]), dtype=float)
    raise ValueError(f"{name} has size {arr.size}, cannot broadcast to {size}")


def index_to_params(linear_index, param_shapes):
    """
    Convert linear index to multi-dimensional parameter indices.

    Special case for filtered data: If param_shapes is [1, 1, ..., 1, N],
    this indicates filtered data where all parameters are aligned at the same index.
    In this case, return [linear_index, linear_index, ..., linear_index].
    """
    n_params = len(param_shapes)
    indices = np.zeros(n_params, dtype=np.int64)

    # Check if this is filtered data (all shapes are 1 except possibly the last)
    if n_params > 1 and np.all(param_shapes[:-1] == 1):
        # Filtered data: all parameters share the same linear index
        indices[:] = linear_index
        return indices

    # Standard grid-based indexing
    remaining = linear_index
    for i in range(n_params - 1, -1, -1):
        indices[i] = remaining % param_shapes[i]
        remaining = remaining // param_shapes[i]

    return indices


def fix_vector_length(vec, target_length=100):
    vec = np.asarray(vec)
    if vec.ndim == 0 or vec.size == 0:
        # If scalar or empty, fill with scalar value or nan
        scalar = float(vec) if vec.size == 1 or vec.ndim == 0 else np.nan
        return np.full(target_length, scalar)
    if vec.size == 1:
        # Repeat the single value
        return np.full(target_length, vec[0])
    if vec.size == target_length:
        return vec
    # Interpolate to target length, preserving first and last value
    x_old = np.linspace(0, 1, vec.size)
    x_new = np.linspace(0, 1, target_length)
    vec_interp = np.interp(x_new, x_old, vec)
    return vec_interp
