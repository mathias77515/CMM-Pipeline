import os
from PIL import Image
import numpy as np
import imageio

def create_folder_if_not_exists(folder_name):
    # Check if the folder exists
    if not os.path.exists(folder_name):
        try:
            # Create the folder if it doesn't exist
            os.makedirs(folder_name)
            #print(f"The folder '{folder_name}' has been created.")
        except OSError as e:
            print(f"Error creating the folder '{folder_name}': {e}")
    else:
        pass


def do_gif(input_folder, filename, output='animation.gif', duration=0.1):

    # Collect all the file paths for the images
    file_paths = []
    for n in sorted(os.listdir(input_folder)):
        if n.startswith(filename) and n.endswith('.png'):
            file_paths.append(os.path.join(input_folder, n))
    
    # Ensure the file_paths are sorted by the numerical part of the filenames
    file_paths = sorted(file_paths, key=lambda x: int(x.split(filename)[-1].split('.png')[0]))

    # Create the GIF
    with imageio.get_writer(os.path.join(input_folder, output), mode='I', duration=duration) as writer:
        for file_path in file_paths:
            image = imageio.imread(file_path)
            writer.append_data(image)

    print(f"GIF saved at {os.path.join(input_folder, output)}")
    
    #N = len(os.listdir(input_folder))
    #nmaps = np.arange(1, N, 1)
    #image_list = []
    #for filename in sorted(os.listdir(input_folder)):
    #for n in nmaps:
    #    image_path = os.path.join(input_folder, filename+f'{n}.png')
    #    image = Image.open(image_path)
    #    image_list.append(image)

    #output_gif_path = os.path.join(input_folder, output)
    #output_gif_path = f"figures/{stk}/animation.gif"
    #image_list[0].save(output_gif_path, save_all=True, append_images=image_list[1:], duration=100, loop=0)
