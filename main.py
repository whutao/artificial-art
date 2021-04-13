import os
from code.evolutionary_solution import run_evolution

USE_CUDA = True

if __name__ == '__main__':
    source_img_dir = './source_images'
    for image_file in os.listdir(source_img_dir):
        image_path = os.path.join(source_img_dir, image_file)
        try:
            print(f'Executing: {image_path}')
            run_evolution(source_image_path=image_path, USE_CUDA=True)
        except KeyboardInterrupt:
            break
        print()
