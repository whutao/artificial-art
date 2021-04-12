# Standard library imports
from math import sqrt

# Third-party library imports
import numpy as np
from numba import cuda


@cuda.reduce
def kernel_sum_reduce(a, b):
    """
    Array reduction kernel.

    Equivalent to np.sum method.

    Args:
        param1: Target array.

    Returns:
        Sum of a given array.

    Usage example:
        arr = np.random.rand(10000)
        arr_sum = kernel_sum_reduce(arr)
    """
    return a + b


@cuda.jit
def kernel_draw_triangle(
    pts: np.ndarray,
    color: np.ndarray,
    source_img: np.ndarray,
    target_img: np.ndarray,
    result_img: np.ndarray,
    fitness_vector: np.ndarray
):
    """
    CUDA kernel for drawing a triangle on a given image.

    Triangle points should be in the form 
    np.array([[Ax, Ay], [Bx, By], [Cx, Cy]], dtype=int).

    Color should be in the form
    np.array([R, G, B], dtype=np.uint8).

    Source and target images should be in the form
    np.array(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8).

    Result image should be in the form
    np.empty(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8). 

    Fitness vector (that is flattened fitness matrix) 
    should be in the form
    np.empty(shape=(HEIGHT*WIDTH,), dtype=np.float64). 
    For each pixel, the fitness is a ratio of 
    Euclidian distance to the original pixel and 
    the largest Euclidian distance between two pixels. 


    Args:
        param1: Triangle points.
        param2: Color of a triangle.
        param3: Source image.
        param4: Target image.
        param5: Result image.
        param6: Fitness vector.

    Returns:
        Returns nothing. 
        Modifies target image and fitness vector.
    """
    # GPU thread coordinates = pixel coordinates
    Px, Py = cuda.grid(2)
    if target_img.shape[0] <= Px or target_img.shape[1] <= Py:
        return

    Ax, Ay = pts[0]
    Bx, By = pts[1]
    Cx, Cy = pts[2]

    # Draw triangle
    ABC = abs(Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
    ABP = abs(Ax*(By-Py) + Bx*(Py-Ay) + Px*(Ay-By))
    APC = abs(Ax*(Py-Cy) + Px*(Cy-Ay) + Cx*(Ay-Py))
    PBC = abs(Px*(By-Cy) + Bx*(Cy-Py) + Cx*(Py-By))
    if ABC == ABP + APC + PBC:
        result_img[Px, Py, 0] = (color[0] + target_img[Px, Py, 0]) // 2
        result_img[Px, Py, 1] = (color[1] + target_img[Px, Py, 1]) // 2
        result_img[Px, Py, 2] = (color[2] + target_img[Px, Py, 2]) // 2
    else:
        result_img[Px, Py, 0] = target_img[Px, Py, 0]
        result_img[Px, Py, 1] = target_img[Px, Py, 1]
        result_img[Px, Py, 2] = target_img[Px, Py, 2]

    # Update fitness vector
    dr = source_img[Px, Py, 0] - result_img[Px, Py, 0]
    dg = source_img[Px, Py, 1] - result_img[Px, Py, 1]
    db = source_img[Px, Py, 2] - result_img[Px, Py, 2]
    color_difference = sqrt(float(dr*dr + dg*dg + db*db)/195075)
    fitness_vector[source_img.shape[1]*Px + Py] = color_difference


if __name__ == '__main__':
    pass
