# Standard library imports
import time
import os
from functools import wraps

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from colorthief import ColorThief


def timing(func):
    """
    This decorator prints function execution time
    in either of these forms:
        1. func:some_function    took: 132 ns...
        2. func:some_function    took: 98 us...
        3. func:some_function    took: 9.67 ms...
        4. func:some_function    took: 1.58 s...
        5. func:some_function    took: 32 min 13 sec...

    Function name will be truncated if it is longer than 16.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_result = func(*args, **kwargs)
        exec_time = time.time() - start_time

        max_length = 16
        if max_length < len(func.__name__):
            func_name = func.__name__[:max_length-3] + '...'
        else:
            func_name = func.__name__ + ' '*(max_length-len(func.__name__))

        print(f'func:{func_name} ', end='')
        if exec_time*1e6 < 1:
            print(f'took: {int(exec_time*1e12)/1e3} ns...')
        elif exec_time*1e3 < 1:
            print(f'took: {int(exec_time*1e9)/1e3} us...')
        elif exec_time < 1:
            print(f'took: {int(exec_time*1e6)/1e3} ms...')
        elif exec_time < 60:
            print(f'took: {int(exec_time*1e3)/1e3} s...')
        else:
            print(f'took: {int(exec_time) // 60} min ', end='')
            print(f'{int(exec_time) % 60} sec...')

        return func_result
    return wrapper


@timing
def open_image(filepath: str) -> np.ndarray:
    """
    Open an image in RGB format. 
    Extentions '.jpg' and '.png' are allowed.

    Args:
        param1: Image file path.

    Returns:
        Opened image as a NumPy 3D array.

    Raises:
        ValueError: File extention is inappropriate.
    """
    filebasename = os.path.basename(filepath)
    extention = os.path.splitext(filename)[1]
    if extention == '.jpg':
        return plt.imread(filepath)
    elif extention == '.png':
        png_img = Image.open(filepath)
        rgb_im = png_img.convert('RGB')
        return np.asarray(rgb_im)
    else:
        raise ValueError(f'Inappropriate extention in {filebasename}!')


@timing
def get_save_dir_path(source_image_path: str) -> str:
    """
    Compose a path for results using source image path.

    Args:
        param1: Path to source image.

    Returns:
        Path to source image.
    """
    filename = os.path.basename(source_image_path)
    image_name = os.path.splitext(filename)[0]
    return os.path.join(f'results/{int(time.time())}_{image_name}')


@timing
def epoch_snapshot(
    best_candidate: np.ndarray,
    best_fitness: np.float64,
    gen_fitnesses: np.ndarray,
    gen_times: np.ndarray,
    gen_id: int,
    source_image: np.ndarray,
    palette: np.ndarray,
    save_dir: str
) -> None:
    """
    An epoch snapshot includes:
        1. Image of the best-fit individual.
        2. Figure with the source image and best-fit individual.
        3. Best fitness for each generation (is added to csv file).
        4. Execution time of each generation (is added to csv file).
        5. Color pallete as an image (if current epoch is the first one).
    The above files are stored in saving directory.

    Args:
        param1: Best-fit individual.
        param2: Best fitness.
        param3: Best fitness for each generation in epoch.
        param4: Execution time of each generation in epoch.
        param5: Current generation ID.
        param6: Source image as a NumPy array.
        param7: Color palette of the source image.
        param8: Directory for saving results.

    Returns:
        Nothing.
    """
    ratio = source_image.shape[0] / source_image.shape[1]

    if source_image.shape[0] > source_image.shape[1]:
        pic_width = 10
        pic_height = pic_width * ratio
    else:
        pic_height = 10
        pic_width = pic_height / ratio

    fig = plt.figure(figsize=(2*pic_width, pic_height))
    fig.suptitle(f'Generation {gen_id}', fontsize=24)

    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title('Source image', fontdict={'fontsize': 18})
    plt.imshow(source_image)

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title(f'error={best_fitness}', fontdict={'fontsize': 18})
    plt.imshow(best_candidate)

    figure_dir = os.path.join(save_dir, 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    figure_filename = os.path.join(figure_dir, f'figure_G{gen_id}.jpg')
    plt.savefig(figure_filename, bbox_inches='tight')
    plt.close('all')

    candidate_dir = os.path.join(save_dir, 'candidate')
    os.makedirs(candidate_dir, exist_ok=True)
    candidate_filename = os.path.join(
        candidate_dir, f'candidate_G{gen_id}.jpg')
    plt.imsave(candidate_filename, best_candidate)

    other_dir = os.path.join(save_dir, 'other')
    os.makedirs(other_dir, exist_ok=True)
    palette_filename = os.path.join(other_dir, 'palette.jpg')
    if not os.path.exists(palette_filename):
        fig = plt.figure(figsize=(2*palette.shape[0], 5))
        plt.subplot(1, 1, 1)
        plt.axis('off')
        plt.title('Color palette', fontdict={'fontsize': 28})
        plt.imshow([palette])
        plt.savefig(palette_filename, bbox_inches='tight')
        plt.close('all')

    fitness_path = os.path.join(other_dir, 'fitness.csv')
    with open(fitness_path, 'a') as fitness_file:
        for i in range(gen_fitnesses.shape[0]):
            fitness_file.write(
                f'{gen_id-gen_fitnesses.shape[0]+i+1},'
                f'{gen_fitnesses[i]}\n')

    time_path = os.path.join(other_dir, 'time.csv')
    with open(time_path, 'a') as time_file:
        for i in range(gen_times.shape[0]):
            time_file.write(
                f'{gen_id-gen_fitnesses.shape[0]+i+1},'
                f'{gen_times[i]}\n')


@timing
def extract_palette(
    image_path: str,
    max_color_count: int = 20
) -> np.ndarray:
    """
    Retrieve color palette from the target image using ColorThief.

    Color count can be less than the provided upperbound.

    Args:
        param1: Image file path.
        param2: Upperbound for the number of colors.

    Returns:
        A palette in the form
        np.array(shape=(color_count, 3), dtype=np.uint8).
    """
    color_thief = ColorThief(image_path)
    raw_palette = color_thief.get_palette(
        color_count=max_color_count,
        quality=1)
    return np.array([
        np.array(color, dtype=np.uint8)
        for color in raw_palette
    ], dtype=np.uint8)


@timing
def best_fit_canvas(
    source_image: np.ndarray,
    palette: np.ndarray
) -> np.ndarray:
    """
    Given the source image and its color palette. 
    Find the best-fit image of the same size with 
    a background. Background color is taken from 
    the palette.

    Args:
        param1: Source image as a NumPy array.
        param2: Source image color palette.

    Returns:
        An image with best fit background as a NumPy array.
    """
    extended_palette = list(palette) + [np.zeros(3), 255*np.ones(3)]
    best_canvas = np.zeros_like(source_image)
    best_fit = 3*source_image.shape[0]*source_image.shape[1]
    for color in extended_palette:
        canvas = np.empty_like(source_image)
        for channel in range(3):
            canvas[:, :, channel].fill(color[channel])

        delta = np.sum((source_image - canvas)**2 / 255**2)
        if delta < best_fit:
            best_fit = delta
            best_canvas = canvas

    return best_canvas
