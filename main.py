if __name__ == '__main__':
    import os
    from code.evolutionary_solution import run_evolution

    source_img_dir = './source_images'
    for image_file in os.listdir(source_img_dir):
        image_path = os.path.join(source_img_dir, image_file)
        try:
            print(f'Executing: {image_path}')
            run_evolution(
                source_image_path=image_path,
                epoch_count=12,
                USE_CUDA=True)
            print()
        except KeyboardInterrupt:
            break
