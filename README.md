# Assignemnt II on Introduction to Artificial Intelligence course

## Source code

Python source code can be found in `code/` folder. It is provided with comments and docstrings. Before running the code, one should: 

1. Have python 3.8+ version installed on their machine.
2. Install the dependencies using `pip3 install -r requirements.txt`.
3. Have CUDA 10+ toolkit installed (optional and only for NVidia GPU users).

Next, one should try to run `main.py` that will start processing test images. Test batch contatins 14 images, and each of them will take about 40-90 minutes. In order to stop the execution, one should use keyboard interrupt `ctrl+c` in their terminal.

If one has no CUDA toolkit installed then `USE_CUDA` flag should be set as `False` in the `main.py`. Image processing will be done by CPU only and hence the execution of a single image will take more than 6 hours.

Generated images can be found in the `results/` folder.

## Report

Detailed report can be found in `report/` folder.

