import numpy as np

def multiprod(a, b, idA=None, idB=None):
    """
    Generalized matrix multiplication of blocks within multi-dimensional arrays.

    Parameters:
    - a: ndarray
    - b: ndarray
    - idA: list of integers (axes for internal dimensions of `a`)
    - idB: list of integers (axes for internal dimensions of `b`)

    Returns:
    - c: result of block-wise multiplications of a and b
    """
    # Default axis setup
    if idA is None and idB is None:
        idA = [0, 1]
        idB = [0, 1]
    elif idB is None:
        idB = idA

    # If both are 2D and default IDs
    if a.ndim == 2 and b.ndim == 2 and idA == [0, 1] and idB == [0, 1]:
        return a @ b

    # Move block dimensions to end of 'a' and front of 'b'
    a_axes = list(range(a.ndim))
    b_axes = list(range(b.ndim))

    a_outer = [ax for ax in a_axes if ax not in idA]
    b_outer = [ax for ax in b_axes if ax not in idB]

    a_perm = a_outer + idA
    b_perm = idB + b_outer

    a_trans = np.transpose(a, a_perm)
    b_trans = np.transpose(b, b_perm)

    a_shape = a_trans.shape
    b_shape = b_trans.shape

    a_batch_shape = a_shape[:-2]
    b_batch_shape = b_shape[2:]

    out_batch_shape = np.broadcast_shapes(a_batch_shape, b_batch_shape)

    a_broadcast = np.broadcast_to(a_trans, out_batch_shape + a_shape[-2:])
    b_broadcast = np.broadcast_to(b_trans, b_shape[:2] + out_batch_shape)

    # Move b's inner dimensions to the back
    b_broadcast = np.moveaxis(b_broadcast, (0, 1), (-2, -1))

    # Matrix multiply along last two dims
    result = np.matmul(a_broadcast, b_broadcast)

    return result