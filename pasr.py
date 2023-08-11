#!/usr/bin/env python3
#
# Author: Alex J. Noble with help from GPT4, June-August 2023 @SEMC
# Associated manuscript: https://doi.org/10.1101/2023.06.09.544325
#
# This script applies the PASR algorithm to 2D images or image stacks.
# Requirement: pip install imagecodecs imageio mrcfile pillow termcolor tifffile
# Usage, single-file PASR: ./pasr.py input.mrc
# Usage, PASRing a folder of mrc files to tif: ./pasr.py images/ -o PASRed/ --force_tif
# Usage, PASRing all tif files in the current folder to JPG100: ./pasr.py *tif -j -q 100

import os
import glob
import time
import mrcfile
import argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from termcolor import colored
from tifffile import imread, imwrite, TiffFile
from multiprocessing import Pool
Image.MAX_IMAGE_PIXELS = None

# Function to scale image
def scale_image(image, scale):
    """Scales an image by duplicating each pixel.

    Args:
    image : numpy array representing the image
    scale : int, scale factor

    Returns:
    numpy array, the scaled image
    """
    return np.repeat(np.repeat(image, scale, axis=1), scale, axis=0)

# Function to process each file
def process_file(input_file, output_file, scale, compression, flip_tif, jpg_quality):
    """Processes a file: reads, scales and writes it.

    Args:
    input_file : str, path to the input file
    output_file : str, path to the output file
    scale : int, scale factor
    compression : str, compression method
    flip_tif : bool, whether to flip the image

    Returns:
    None
    """
    try:
        # Check if output is JPG and read header to check dimensions
        if output_file.endswith('.jpg'):
            if input_file.endswith('.mrc') or input_file.endswith('.mrcs'):
                with mrcfile.mmap(input_file, mode='r') as mrc:
                    if mrc.header['nz'] > 1:
                        print(colored(f'Error: Cannot save 2D frame stack as JPG for {input_file}', 'red'))
                        return
            elif input_file.endswith('.tif') or input_file.endswith('.tiff'):
                with TiffFile(input_file) as tif:
                    original_data_dim = len(tif.series[0].shape)
                    if original_data_dim > 2:
                        print(colored(f'Error: Cannot save 2D frame stack as JPG for {input_file}', 'red'))
                        return

        # Reading the file
        if input_file.endswith('.mrc') or input_file.endswith('.mrcs'):
            with mrcfile.open(input_file) as mrc:
                original_data = mrc.data
        elif input_file.endswith('.tif') or input_file.endswith('.tiff'):
            original_data = imread(input_file)
        elif input_file.endswith('.jpg') or input_file.endswith('.jpeg'):
            original_data = imageio.imread(input_file)
        else:
            raise ValueError(colored(f'Unsupported input file format for {input_file}. Only MRC, TIF and JPG files are supported.', 'red'))

        # Scaling the image
        if original_data.ndim == 3:
            processed_data = np.empty((original_data.shape[0], original_data.shape[1]*scale, original_data.shape[2]*scale), dtype=original_data.dtype)
            for i in range(original_data.shape[0]):
                processed_data[i] = scale_image(original_data[i], scale)
        elif original_data.ndim == 2:
            processed_data = scale_image(original_data, scale)
        else:
            raise ValueError(colored(f'Unsupported data dimensions for {input_file}. Only 2D images and 2D frame stacks are supported.', 'red'))

        # Writing the output
        if output_file.endswith('.mrc') or output_file.endswith('.mrcs'):
            with mrcfile.new(output_file, overwrite=True) as mrc:
                mrc.set_data(processed_data)
            print(colored(f'PASR pre-processed data written to {output_file}. No compression applied for MRC format (use TIF for compression).', 'green'))
        elif output_file.endswith('.tif') or output_file.endswith('.tiff'):
            if flip_tif:
                processed_data = np.flip(processed_data, axis=0)
                flip_status = "Flipping applied."
            else:
                flip_status = "No flipping applied."
            imwrite(output_file, processed_data, compression=compression)
            print(colored(f'PASR pre-processed data written to {output_file} with {compression} compression. {flip_status}', 'green'))
        elif output_file.endswith('.jpg'):
            processed_data = processed_data.astype(np.uint8)
            im = Image.fromarray(processed_data)
            base_name, ext = os.path.splitext(output_file)
            output_file = f"{base_name}_JPG{jpg_quality}{ext}"
            im.save(output_file, quality=jpg_quality)
            print(colored(f'PASR pre-processed data written to {output_file} with JPEG quality={jpg_quality}', 'green'))
        else:
            raise ValueError(colored(f'Unsupported output file format for {input_file}. Only MRC, TIF, and JPG for 2D images are supported.', 'red'))
    except Exception as e:
        print(colored(f'An error occurred while processing file {input_file}: {e}', 'red'))

# Function to process a directory
def process_files(file_list, output_dir, scale, compression, flip_tif, force_tif, force_mrc, force_jpg, n_cores, keep_basename, jpg_quality):
    """Processes a directory: lists all files and processes each of them.

    Args:
    input_dir : str, path to the input directory
    output_dir : str, path to the output directory
    scale : int, scale factor
    compression : str, compression method
    flip_tif : bool, whether to flip the image
    force_tif : bool, whether to force TIF output
    force_mrc : bool, whether to force MRC output
    force_jpg : bool, whether to force JPG output
    n_cores : int, number of CPU cores to use
    keep_basename : bool, whether to keep the original file basename

    Returns:
    None
    """
    # Filter files by supported extensions
    supported_extensions = ['.mrc', '.mrcs', '.tif', '.tiff', '.jpg', '.jpeg']
    file_list = [file for file in file_list if os.path.splitext(file)[1].lower() in supported_extensions]

    # Get the number of files and the minimum of cores and number of files
    num_files = len(file_list)
    min_cores = min(n_cores, num_files)
    
    # Get a set of unique file types in the directory
    file_types = set(os.path.splitext(file)[1][1:] for file in file_list)
    input_types = ', '.join(file_type.upper() for file_type in file_types)
    
    print(colored(f'PASR-processing {num_files} {input_types} files across {min_cores} CPU cores...', 'green'))

    # Creating the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Creating a list of arguments for multiprocessing
    arguments = []
    for input_file in file_list:
        base_name, ext = os.path.splitext(os.path.basename(input_file))
        if force_tif:
            if keep_basename:
                output_file = os.path.join(output_dir, f"{base_name}.tif")
            else:
                output_file = os.path.join(output_dir, f"{base_name}_PASR_{scale}x.tif")
        elif force_mrc:
            if keep_basename:
                output_file = os.path.join(output_dir, f"{base_name}.mrc")
            else:
                output_file = os.path.join(output_dir, f"{base_name}_PASR_{scale}x.mrc")
        elif force_jpg:
            if keep_basename:
                output_file = os.path.join(output_dir, f"{base_name}.jpg")
            else:
                output_file = os.path.join(output_dir, f"{base_name}_PASR_{scale}x.jpg")
        elif keep_basename:
            output_file = os.path.join(output_dir, f"{base_name}{ext}")
        else:
            output_file = os.path.join(output_dir, f"{base_name}_PASR_{scale}x{ext}")
        if flip_tif is None:
            flip_tif = input_file.lower().endswith(('.mrc', '.mrcs')) and output_file.lower().endswith(('.tif', '.tiff'))
        arguments.append((input_file, output_file, scale, compression, flip_tif, jpg_quality))

    # Using a multiprocessing pool to process all files in parallel
    with Pool(n_cores) as p:
        p.starmap(process_file, arguments)
    
    return num_files

def main():
    start_time = time.time()
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="PASR Pre-process MRC and TIF frame stacks and MRC, TIF, and JPG images.")
    parser.add_argument("input", type=str, nargs="+", help="Input MRC, MRCS, TIF, TIFF, or JPG file, or directory containing such files, or a list of such files.")
    parser.add_argument("-s", "--scale", type=int, choices=range(2, 6), default=2, help="Scaling factor (number of times to duplicate each pixel)")
    parser.add_argument("-o", "--output", type=str, help="Output MRC, TIF, or TIFF file, or directory to store processed files. If not provided and the input is a directory, the input directory is used.")
    parser.add_argument("-c", "--compression", type=str, choices=['zlib', 'lzw'], default='lzw', help="Compression algorithm for TIF output. ZLIB is smaller, but may not work with some software.")
    parser.add_argument("-q", "--jpg_quality", type=int, default=95, choices=range(1, 101), help="Quality for JPG output, from 1 (worst) to 100 (best). Note: values above 95 may not increase quality much, but will increase file size substantially. Default is 95.")
    parser.add_argument("-n", "--n_cores", type=int, default=os.cpu_count(), help="Number of CPU cores to use for parallel processing.")
    parser.add_argument("-k", "--keep_basename", action="store_true", help="Keep the original basename for the output file(s); do not append _PASR_{scale}x.")
    parser.add_argument("-f", "--flip_tif", type=lambda x: (str(x).lower() in ['true','1', 'yes']), nargs='?', const=True, default=None, help="Option to flip TIF output across the x-axis. Specify --flip_tif=True to enable, --flip_tif=False to disable. Default is True for MRC/MRCS input and TIF/TIFF output, False otherwise, unless specified.")
    parser.add_argument("-t", "--tif", "--force_tif", action="store_true", help="Force the output file extension to be .tif. This is useful because .tif output uses ZLIB or LZW compression.")
    parser.add_argument("-m", "--mrc", "--force_mrc", action="store_true", help="Force the output file extension to be .mrc. This exists just for completion.")
    parser.add_argument("-j", "--jpg", "--force_jpg", action="store_true", help="Force the output file extension to be .jpg for 2D images. This exists just for fun.")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically answer 'y' to any user input questions.")
    args = parser.parse_args()

    # Check that a maximum of one 'force' option is used
    if sum([args.tif, args.mrc, args.jpg]) > 1:
        print(colored("Error: Only one of the 'force' options (--force_tif, --force_mrc, --force_jpg) can be used at a time.", 'red'))
        exit()

    # Check if the input is a list of files
    if len(args.input) > 1:
        # If the output path is not specified, set it to the input directory
        if args.output is None:
            args.output = os.path.dirname(os.path.abspath(args.input[0]))
        # If the output directory does not exist, create it
        elif not os.path.isdir(args.output):
            os.makedirs(args.output)

        # Add user confirmation for potential file overwriting
        if os.path.normpath(os.path.dirname(os.path.abspath(args.input[0]))) == os.path.normpath(args.output) and not args.yes:
            confirm = input(colored(f'Warning: Your input and output directories are the same ({args.output}). This may overwrite your files. Continue? [y/N]', 'yellow'))
            if confirm.lower() != 'y':
                exit()
        num_files = process_files(args.input, args.output, args.scale, args.compression, args.flip_tif, args.tif, args.mrc, args.jpg, args.n_cores, args.keep_basename, args.jpg_quality)

    # Check if the input is a directory
    elif os.path.isdir(args.input[0]):
        input_path = args.input[0]
        # If the output path is not specified, set it to the input directory
        if args.output is None:
            args.output = input_path
        # If the output directory does not exist, create it
        elif not os.path.isdir(args.output):
            os.makedirs(args.output)

        # Add user confirmation for potential file overwriting
        if os.path.normpath(input_path) == os.path.normpath(args.output) and not args.yes:
            confirm = input(colored(f'Warning: Your input and output directories are the same ({args.output}). This may overwrite your files. Continue? [y/N]', 'yellow'))
            if confirm.lower() != 'y':
                exit()

        file_list = glob.glob(os.path.join(input_path, "*"))
        num_files = process_files(file_list, args.output, args.scale, args.compression, args.flip_tif, args.tif, args.mrc, args.jpg, args.n_cores, args.keep_basename, args.jpg_quality)
    
    # Check if the input is one file
    elif len(args.input) == 1:
        input_path = args.input[0]
        num_files = 1
        # If the output file is not specified, we create one with the same name as the input, but append "_PASR" before the extension
        if args.output is None:
            base_name, ext = os.path.splitext(input_path)
            if args.keep_basename:
                args.output = f"{base_name}{ext}"
            else:
                args.output = f"{base_name}_PASR_{args.scale}x{ext}"

        # Overwrite file extension if necessary
        if args.tif:
            base_name, ext = os.path.splitext(args.output)
            args.output = f"{base_name}.tif"
        if args.mrc:
            base_name, ext = os.path.splitext(args.output)
            args.output = f"{base_name}.mrc"
        if args.jpg:
            base_name, ext = os.path.splitext(args.output)
            args.output = f"{base_name}_JPG{args.jpg_quality}.jpg"

        # Determine if the output should be flipped
        if args.flip_tif is None:
            args.flip_tif = input_path.lower().endswith(('.mrc', '.mrcs')) and args.output.lower().endswith(('.tif', '.tiff'))

        # Add user confirmation for potential file overwriting
        if os.path.exists(args.output) and not args.yes:
            confirm = input(colored(f'Warning: Your output file already exists ({args.output}). This will overwrite your file. Continue? [y/N]', 'yellow'))
            if confirm.lower() != 'y':
                print(colored(f'Skipping file {input_path}', 'yellow'))
                exit()
        if args.jpg:
            args.output = f"{base_name}.jpg" # The previous change was just for comparison. This reverts back, then it's changed again in the process_file function.

        process_file(input_path, args.output, args.scale, args.compression, args.flip_tif, args.jpg_quality)

    end_time = time.time()
    print(f"Total time taken to process {num_files} file(s): {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()