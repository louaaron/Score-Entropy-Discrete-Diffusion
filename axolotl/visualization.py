import os
import gc
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import imageio
from PIL import Image
from multiprocessing import Pool

def save_logo_plot_wrapper(args):
    try:
        # Your existing code here
        return save_logo_plot(*args)
    except Exception as e:
        print(f"An error occurred in worker process: {e}")
        traceback.print_exc()  # This will print the stack trace

def save_logo_plot(array, label:str, png_dir_str:str, positions_per_line:int, width:int = 100, ylim:tuple = (-1,3), dpi:int = 100, characters:str = "ACDEFGHIKLMNPQRSTVWY?[]-"):
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got an array with shape {array.shape}. Please provide a 2D array.")

    if not os.path.exists(png_dir_str):
        os.makedirs(png_dir_str)

    amino_acids = list(characters)

    if amino_acids[-1] == '-':
        amino_acids = amino_acids[:-1]
        array = array[:-1, :]
    
    png_path = png_dir_str + '/' + f"sequence_logo_{label}.png"

    if os.path.exists(png_path): # If the file already exists, return the path.
        return png_path

    num_positions = array.shape[1]
    num_lines = (num_positions + positions_per_line - 1) // positions_per_line
    
    fig, axes = plt.subplots(num_lines, 1, figsize=(width, 5 * num_lines), squeeze=False)
    
    for line in range(num_lines):
        start = line * positions_per_line
        end = min(start + positions_per_line, num_positions)
        df = pd.DataFrame(array.T[start:end], columns=amino_acids)
        
        logo = logomaker.Logo(df, ax=axes[line, 0])
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        logo.ax.set_ylabel("Probability")
        logo.ax.set_xlabel("Position")
        logo.ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.title(f"Sequence Logo for Tensor: {label}")

    # Save the figure as a PNG file
    plt.savefig(png_path, dpi = dpi)
    plt.close(logo.fig)
    plt.close(fig)
    del logo
    del axes
    del fig
    gc.collect()  # Force garbage collection
    
    return png_path

def plot_sequence_logo_and_create_gif(tensor_cpu_numpy, positions_per_line, ylim = (-1,3), dpi = 100, characters="ACDEFGHIKLMNPQRSTVWY?[]-", output_gif_path="sequence_logos.gif", png_dir = "sequence_logo_pngs", num_processes = 10):

    """
    Plots sequence logos from a tensor and creates a GIF from the saved PNG files.
    Args:
        tensor_cpu_numpy (nd.array): A numpy array of shape (batch_size, timestep, num_amino_acids, sequence_length) containing the sequence data.
    """

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # Create a multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        # Prepare the arguments for each function call
        def args_generator():
            for idx, array in enumerate(tensor_cpu_numpy):
                yield (array, idx, str(png_dir), positions_per_line, positions_per_line, ylim, dpi, characters)

        # Use map to apply the function to the arguments in parallel
        png_files = pool.map(save_logo_plot_wrapper, args_generator())
        
    # Create a GIF from the saved PNG files
    with imageio.get_writer(output_gif_path, mode='I', duration=2) as writer:
        for png_file in png_files:
            try:
                image = Image.open(png_file)
                writer.append_data(np.array(image))
                print(f"Appended image {png_file} to GIF")
            except Exception as e:
                print(f"Error opening and appending image {png_file} to GIF: {e}")
            image.close()
            del image
            gc.collect()  # Force garbage collection

    print(f"GIF saved at {output_gif_path}")
