if __name__ == '__main__':
    from code.evolutionary_solution import run_evolution

    image_files = [
        'source_images/at the hockey - resize.jpg',
        'source_images/head.jpg']

    for image_file in image_files:
        try:
            print(f'Executing: {image_file}')
            run_evolution(source_image_path=image_file, USE_CUDA=True)
            print()
        except KeyboardInterrupt:
            pass
