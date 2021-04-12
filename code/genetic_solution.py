# Standard library imports
import time

# Third-party library imports
import numpy as np
from numba import cuda

# Local imports
from code.helper import timing
from code.helper import open_image
from code.helper import get_save_dir_path
from code.helper import epoch_snapshot
from code.helper import extract_palette
from code.helper import best_fit_canvas
from code.cuda_kernels import kernel_draw_triangle
from code.cuda_kernels import kernel_sum_reduce


def mutation(
    chromosome: np.ndarray,
    source_image: np.ndarray,
    palette: np.ndarray,
    height: int,
    width: int,
    image_padding: int,
    frame_height: int,
    frame_width: int,
    TPB: (int, int),
    BPG: (int, int)
) -> (np.ndarray, np.ndarray):
    padd = image_padding
    fsh = int(height*np.random.rand()) - padd
    fsw = int(width*np.random.rand()) - padd

    pts = np.empty(shape=(3, 2), dtype=np.int64)
    pts[0, 0] = int(fsh + (frame_height + 2*padd)*np.random.rand() - padd)
    pts[0, 1] = int(fsw + (frame_width + 2*padd)*np.random.rand() - padd)
    pts[1, 0] = int(fsh + (frame_height + 2*padd)*np.random.rand() - padd)
    pts[1, 1] = int(fsw + (frame_width + 2*padd)*np.random.rand() - padd)
    pts[2, 0] = int(fsh + (frame_height + 2*padd)*np.random.rand() - padd)
    pts[2, 1] = int(fsw + (frame_width + 2*padd)*np.random.rand() - padd)

    idx = np.random.randint(palette.shape[0])
    color = palette[idx]

    pts_global_mem = cuda.to_device(pts)
    color_global_mem = cuda.to_device(color)
    source_img_global_mem = cuda.to_device(source_image)
    target_img_global_mem = cuda.to_device(chromosome)
    result_image_global_mem = cuda.device_array(
        shape=(height, width, 3), dtype=np.uint8)
    fitness_vector_global_mem = cuda.device_array(
        shape=(height*width,), dtype=np.float64)

    kernel_draw_triangle[BPG, TPB](
        pts_global_mem,
        color_global_mem,
        source_img_global_mem,
        target_img_global_mem,
        result_image_global_mem,
        fitness_vector_global_mem)

    new_chromosome = result_image_global_mem.copy_to_host()
    new_fitness = kernel_sum_reduce(fitness_vector_global_mem)

    return new_chromosome, new_fitness


def run_generation(
    current_gen: np.ndarray,
    current_fitnesses: np.ndarray,
    population_size: int,
    source_image: np.ndarray,
    palette: np.ndarray,
    height: int,
    width: int,
    image_padding: int,
    frame_height: int,
    frame_width: int,
    TPB: (int, int),
    BPG: (int, int)
) -> (np.ndarray, np.ndarray):
    next_gen = np.empty_like(current_gen)
    next_fitnesses = np.empty_like(current_fitnesses)

    # Select the best-fit candidate
    best_candidate = current_gen[0]
    best_fitness = current_fitnesses[0]
    for i in range(1, population_size):
        if current_fitnesses[i] < best_fitness:
            best_candidate = current_gen[i]
            best_fitness = current_fitnesses[i]
    next_gen[0] = best_candidate
    next_fitnesses[0] = best_fitness

    # Calculate weights for selection
    p = (1.0/current_fitnesses) / np.sum(1.0/current_fitnesses)

    # Fill the next generation
    for i in range(1, population_size):
        # Selection
        idx = np.random.choice(population_size, p=p)
        successor, successor_fitness = mutation(
            chromosome=current_gen[idx],
            source_image=source_image,
            palette=palette,
            height=height,
            width=width,
            image_padding=image_padding,
            frame_height=frame_height,
            frame_width=frame_width,
            TPB=TPB,
            BPG=BPG)

        next_gen[i] = successor
        next_fitnesses[i] = successor_fitness

    return next_gen, next_fitnesses


@timing
def run_epoch(
    current_gen: np.ndarray,
    current_fitnesses: np.ndarray,
    epoch_duration: int,
    population_size: int,
    source_image: np.ndarray,
    palette: np.ndarray,
    height: int,
    width: int,
    image_padding: int,
    frame_height: int,
    frame_width: int,
    TPB: (int, int),
    BPG: (int, int)
) -> (np.ndarray, np.ndarray):
    gen_fitnesses = np.empty(shape=(epoch_duration,), dtype=np.float64)
    gen_times = np.empty(shape=(epoch_duration,), dtype=np.float64)
    for i in range(epoch_duration):
        current_gen, current_fitnesses = run_generation(
            current_gen=current_gen,
            current_fitnesses=current_fitnesses,
            population_size=population_size,
            source_image=source_image,
            palette=palette,
            height=height,
            width=width,
            image_padding=image_padding,
            frame_height=frame_height,
            frame_width=frame_width,
            TPB=TPB,
            BPG=BPG)

        gen_fitnesses[i] = current_fitnesses[0]
        gen_times[i] = time.time()

    return current_gen, current_fitnesses, gen_fitnesses, gen_times


@timing
def run_evolution(
    source_image_path: str,
    save_dir_path: str = None,
    epoch_duration: int = None,
    epoch_count: int = None,
    fitness_limit: float = None,
    population_size: int = None,
    canvas: np.ndarray = None
) -> None:
    source_image = open_image(source_image_path)
    height, width, _ = source_image.shape
    palette = extract_palette(source_image_path)
    image_padding = min(height, width) // 36
    # image_padding = 10
    frame_height, frame_width = (image_padding, image_padding)

    TPB = (16, 16)
    BPG = (int(np.ceil(height/TPB[0])), int(np.ceil(width/TPB[1])))

    if save_dir_path is None:
        save_dir = get_save_dir_path(source_image_path)

    if epoch_duration is None:
        epoch_duration = 10000

    if epoch_count is None:
        epoch_count = 16

    if fitness_limit is None:
        fitness_limit = 0.05*height*width

    if population_size is None:
        population_size = 8

    if canvas is None:
        canvas = best_fit_canvas(
            source_image=source_image,
            palette=palette)

    population = np.array([
        canvas for _ in range(population_size)
    ], dtype=np.uint8)

    fitnesses = np.repeat(float(height*width*3), population_size)

    epoch_snapshot(
        best_candidate=population[0],
        best_fitness=fitnesses[0],
        gen_fitnesses=np.array([fitnesses[0]]),
        gen_times=np.array([time.time()]),
        gen_id=0,
        source_image=source_image,
        palette=palette,
        save_dir=save_dir)

    for epoch_id in range(1, epoch_count+1):
        population, fitnesses, gen_fitnesses, gen_times = run_epoch(
            current_gen=population,
            current_fitnesses=fitnesses,
            epoch_duration=epoch_duration,
            population_size=population_size,
            source_image=source_image,
            palette=palette,
            height=height,
            width=width,
            image_padding=image_padding,
            frame_height=frame_height,
            frame_width=frame_width,
            TPB=TPB,
            BPG=BPG)

        best_candidate = population[0]
        best_fitness = fitnesses[0]
        for i in range(1, population_size):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_candidate = population[i]

        epoch_snapshot(
            best_candidate=best_candidate,
            best_fitness=best_fitness,
            gen_fitnesses=gen_fitnesses,
            gen_times=gen_times,
            gen_id=epoch_id*epoch_duration,
            source_image=source_image,
            palette=palette,
            save_dir=save_dir)

        if best_fitness < fitness_limit:
            return


if __name__ == '__main__':
    pass
