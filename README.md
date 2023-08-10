# PASR Pre-Processing Script

This script applies the PASR (Post-Acquisition Super Resolution) algorithm to 2D images or image stacks. The script is designed to handle files in MRC, TIF, TIFF, and JPG formats, and can process individual files, a directory of files, or a list of files. The PASR algorithm is a type of image pre-processing that scales images by duplicating each pixel a specified number of times. The result is that cryoEM/cryoET particle alignments that reach Nyquist resolution before PASR mayreach a beyond-Nyquist resolution after PASR. This is possible due to super-sampling and sub-pixel cross-correlation accuracy during frame alignment and/or particle alignment steps.

## Installation

This script requires Python 3 and several Python libraries. These libraries can be installed via pip:

```bash
pip install mrcfile tifffile imagecodecs termcolor pillow
```

## Usage

The script can be run from the command line and takes a number of arguments.

To process a single file:
```bash
./pasr.py input.mrc
```

To process a directory of files:
```bash
./pasr.py images/ -o PASRed/ --force_tif
```

To process all files in the current directory:
```bash
./pasr.py `` `*` ``
```

## Arguments

- `input`: The path to the input file(s) or directory, or a list of files.
- `-s`, `--scale`: The scaling factor (i.e., the number of times to duplicate each pixel). Choices are from 2 to 5. Default is 2.
- `-o`, `--output`: The path to the output file or directory. If not provided and the input is a directory, the input directory is used.
- `-c`, `--compression`: The compression algorithm to use for TIF output. Choices are 'zlib' and 'lzw'. Default is 'lzw'.
- `-q`, `--jpg_quality`: The quality for JPG output, from 1 (worst) to 100 (best). Note: values above 95 may not increase quality much but will increase file size substantially. Default is 95.
- `-n`, `--n_cores`: The number of CPU cores to use for parallel processing. Default is the number of cores available on the system.
- `-k`, `--keep_basename`: Keep the original basename for the output file(s); do not append `_PASR_{scale}x`.
- `-f`, `--flip_tif`: Option to flip TIF output across the x-axis. Default is True for MRC/MRCS input and TIF/TIFF output, False otherwise, unless specified.
- `-t`, `--tif`, `--force_tif`: Force the output file extension to be .tif. This is useful because .tif output uses ZLIB or LZW compression.
- `-m`, `--mrc`, `--force_mrc`: Force the output file extension to be .mrc. This exists just for completion.
- `-j`, `--jpg`, `--force_jpg`: Force the output file extension to be .jpg for 2D images. This exists just for fun.
- `-y`, `--yes`: Automatically answer 'y' to any user input questions, such as overwriting warnings.

## Details

The script is parallelized for efficiency when processing multiple files using as many CPU cores as are available on the system, unless specified otherwise.

Recommended output formats are TIF (best lossless compression) or MRC. JPEG output is also an option for fun development and archival purposes.

## Example with Options

Here is an example command using several of the available options:

```bash
./pasr.py images/ -o PASRed/ -s 3 -c zlib -n 4 --force_tif
```

This command will do the following:

- `images/`: Process all files in the `images/` directory.
- `-o PASRed/`: Write the processed files to the `PASRed/` directory.
- `-s 3`: Use a scale factor of 3, meaning each pixel in the original images will be replaced with a 3x3 grid of pixels in the processed images. Note: Only do this if you have already reached Nyquist for -s 2.
- `-c zlib`: Use zlib compression for TIF output.
- `-n 4`: Use 4 CPU cores for parallel processing.
- `--force_tif`: Save all output files as .tif, regardless of the input file format.

This is just an example. Adjust the command to suit your needs!

## Reference

For more details about the PASR algorithm and its applications, see the associated manuscript by Burton-Smith & Murata, 2023: [https://doi.org/10.1101/2023.06.09.544325](https://doi.org/10.1101/2023.06.09.544325)

## Author

This script was written by Alex J. Noble with assistance from OpenAI's GPT-4 model, June-August 2023.
