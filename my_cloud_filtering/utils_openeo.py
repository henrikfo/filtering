import os 
import io
import tarfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio.plot

def save_image_result(image_data, is_ndvi=False, output_folder='saved_images', output_filename=None, filename_prefix="png"):

    try:
        return save_single_image(image_data, is_ndvi, output_folder, output_filename, filename_prefix)
    except Exception as e:
        pass
    return save_multiple_images(image_data, is_ndvi, output_folder, output_filename, filename_prefix)

def save_single_image(image_data, is_ndvi=False, output_folder='saved_images', output_filename=None, filename_prefix=None):
    """
    Saves the image data to a file instead of displaying it.

    Parameters:
        image_data (bytes): Byte data of an image file (e.g., from openeo).
        is_ndvi (bool): Set to True if the image data is an NDVI image to use an appropriate color map.
        output_folder (str): Directory where the image file will be saved.
        output_filename (str): Name of the file to save the image. If None, uses a timestamp from image metadata.
        filename_prefix (str): Prefix for the filenames. If None, uses timestamps from image metadata.

    Returns:
        str: Path to the saved image file.
    """
    # Create a file-like object from byte data
    filelike = io.BytesIO(image_data)

    # Open the image file using rasterio
    with rasterio.open(filelike) as im:
        fig, ax = plt.subplots()
        
        # Set color map based on whether it is an NDVI image
        if is_ndvi:
            show_params = {'cmap': 'RdYlGn', 'vmin': -0.8, 'vmax': 0.8}
        else:
            show_params = {'cmap': 'pink'}
        
        print(im.__dir__())
        
        rasterio.plot.show(im, ax=ax, **show_params)

        # Get the datetime from the image metadata to use as a default filename
        if output_filename is None:
            datetime_str = im.tags()["datetime_from_dim"].replace(":", "").replace("-", "").replace(" ", "_")
            output_filename = f"{datetime_str}.{filename_prefix}"

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Full path for the output file
        full_path = os.path.join(output_folder, output_filename)

        # Save the figure to a file
        plt.savefig(full_path, dpi=1200)
        plt.close(fig)

    return output_folder, [output_filename]

def save_multiple_images(image_data, is_ndvi=False, output_folder='saved_images', output_filename=None, filename_prefix="png"):
    """
    Saves multiple images extracted from a compressed byte stream to files.

    Parameters:
        image_data (bytes): Byte data of a compressed file containing multiple image files (e.g., tar.gz).
        is_ndvi (bool): Set to True if the images are NDVI images to use an appropriate color map.
        output_folder (str): Directory where the image files will be saved.
        output_filename (str): Name of the file to save the image. If None, uses a timestamp from image metadata.
        filename_prefix (str): Prefix for the filenames. If None, uses timestamps from image metadata.

    Returns:
        list of str: Paths to the saved image files.
    """
    # Create a list to store the paths of saved images
    saved_image_paths = []

    # Create a file-like object from byte data
    file_like_object = io.BytesIO(image_data)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Open the tar.gz file
        with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
            # Extract all the contents into the temporary directory
            tar.extractall(tmpdirname)
            
            # Assume the first directory inside is where images are located (if nested)
            if os.path.isdir(tmpdirname):
                tmpdirname = f"{tmpdirname}/{os.listdir(tmpdirname)[0]}"
            
            # Identify TIFF files in the directory
            ifnames = [ifname for ifname in sorted(os.listdir(tmpdirname))
                       if ifname.endswith(".tif")]
            
            for idx, ifname in enumerate(ifnames):
                image_path = os.path.join(tmpdirname, ifname)
                # Open the image using rasterio
                with rasterio.open(image_path) as im:
                    fig, ax = plt.subplots()
                    
                    # Set color map based on whether it is an NDVI image
                    if is_ndvi:
                        show_params = {'cmap': 'RdYlGn', 'vmin': -0.8, 'vmax': 0.8}
                    else:
                        show_params = {'cmap': 'pink'}
                    
                    # Display the image
                    rasterio.plot.show(im, ax=ax, **show_params)
                    
                    # # Generate filename using metadata or prefix
                    # datetime_str = im.tags().get("datetime_from_dim", "unknown").replace(":", "").replace("-", "").replace(" ", "_")
                    
                    if output_filename is None:
                        datetime_str = im.tags()["datetime_from_dim"].replace(":", "").replace("-", "").replace(" ", "_")
                        output_filename = f"{datetime_str}"

                    # output_filename = f"{filename_prefix or ''}{datetime_str}.png"
                    
                    # Ensure the output folder exists
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    
                    # Full path for the output file
                    full_path = os.path.join(output_folder, output_filename+f"{idx}.{filename_prefix}")
                    
                    # Save the figure to a file
                    plt.savefig(full_path, dpi=1200)
                    plt.close(fig)
                    
                    # Append the saved file path to the list
                    saved_image_paths.append(full_path)
    
    return output_folder, saved_image_paths