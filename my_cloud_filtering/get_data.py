import matplotlib.pyplot as plt
import numpy as np
import openeo
import os
import io
import rasterio

from datetime import datetime, timedelta

from config import *
from PIL import Image

def pool_bands(data: dict, min_band_dim: int, func=Image.LANCZOS) -> dict:
    # Functions to use: Image.LANCZOS, Image.BILINEAR, ...

    resized_data = []
    for channel in [*data.values()]:
        if not channel.shape[0] == min_band_dim[0]:
            # down_scale_factor = np.round(channel.shape[0] / min_band_dim[0])
            # channel = block_reduce(channel, block_size=(down_scale_factor, down_scale_factor), func=np.mean)
            # channel = channel.resize()
            channel = Image.fromarray(channel)
            channel = channel.resize((min_band_dim[1], min_band_dim[0]), resample=func)
            channel = np.asarray(channel)

        resized_data.append(channel)
    return np.asarray(resized_data)

def interpolate_bands(data: dict, max_band_dim: int, func=Image.LANCZOS) -> dict:
    # Functions to use: Image.LANCZOS, Image.BILINEAR, ...

    resized_data = []
    for channel in [*data.values()]:
        if not channel.shape[0] == max_band_dim[0]:
            # up_scale_factor = int(max_band_dim / channel.shape[0])
            # channel = zoom(channel, (up_scale_factor, up_scale_factor), order=1)
            channel = Image.fromarray(channel)
            channel = channel.resize((max_band_dim[1], max_band_dim[0]), resample=func)
            channel = np.asarray(channel)

        resized_data.append(channel)
    return np.asarray(resized_data)
    

def get_product(data:dict, band_list: list = [], scaling: str = "upsizing", scaling_function=Image.LANCZOS) -> np.ndarray:
    # Scaling methods to use are either upsizing (interpolation e.g. 200x200 -> 400x400)
    #       or downsizing (pooling e.g. 400x400 -> 200x200))

    chosen_band_data = {}
    for band in band_list:
        assert band in list(data.keys()) # Fails if band not found in input data
        chosen_band_data[band] = data[band]

    # Get all dim sizes for all bands
    band_dim = [band.shape for band in chosen_band_data.values()]

    # Scalng method to make all band dimmension-sizes allign
    if scaling == "upsizing":
        array_data = interpolate_bands(chosen_band_data, max(band_dim), func=scaling_function)

    elif scaling == "downsizing":
        array_data = pool_bands(chosen_band_data, min(band_dim), func=scaling_function)
    
    return array_data

def get_l1c_data():
    base_path = "../data/l1c/"
    data = {}

    for i in range(1, 13):

        channel = ("B0"if i < 10 else "B")+str(i)
        filename = "T34WES_20240904T100551_"+channel+".jp2"

        with rasterio.open(base_path+filename) as dataset:
            data[channel] = (dataset.read(1) - 1000) / 10000

    with rasterio.open(base_path+"T34WES_20240904T100551_B8A.jp2") as dataset:
            data["B8A"] = (dataset.read(1) - 1000) / 10000

    return data

def download_eo_data(params) -> list:
    
    l2a_bands_resolutions = {
            "60": ["B01", "b09"],
            "10": ["b02", "b03", "b04", "b08"],
            "20": ["b05", "b06", "b07", "b8a", "b11", "b12", "scl", "cld", "snw", "wvp", "aot"]}

    # Params for openEO data download
    params = params["geojson"]

    date = datetime.strptime(params["time"]["date"], '%Y-%m-%d')
    
    collection= params["collection"]
    collection = "s2_msi_l2a" if collection == "l2a" else "s2_msi_l1c"
    
    bands_params=[band.lower() for band in params["bands"]]
    coords=params["geometry"]["coords"]
    
    # Initialize dictionary to hold bands grouped by resolution
    bands_by_resolution = {"60": [], "10": [], "20": []}

    # Group bands to download by their resolution
    for res, bands in l2a_bands_resolutions.items():
        bands_by_resolution[res] = [band for band in bands if band in bands_params]
    
    image_data = {}
    for _resolution, bands in bands_by_resolution.items():
        # Bands list not empty, try downloading the band data
        if bands:
            try:
                connect = openeo.connect(eo_service_url)
                connect.authenticate_basic(username=user, password=passwd)

                cube = connect.load_collection(
                    collection_id=collection,
                    spatial_extent=coords,
                    temporal_extent=[date, (date + timedelta(days=1))],
                    bands=bands
                )
                downloaded_data = cube.download(format="gtiff")
                for i, band in enumerate(bands):

                    filelike = io.BytesIO(downloaded_data)
                    im = rasterio.open(filelike)
                    timestamp =  im.tags()["timestamp"]

                    if datetime.fromisoformat(timestamp.split("T")[0]) >= datetime(2022, 1, 1): 
                        image_data[band.upper()] = (im.read(i+1) - 1000) / 10000
                    else:
                        image_data[band.upper()] = im.read(i+1) / 10000

            except Exception as e:
                if "Collection can not be found with the given parameters" in str(e):
                    print("Could not find data for that date/area")
                return

    return image_data

def get_l2a_data(params):
    return download_eo_data(params)

def get_data(source: str = "l1c", date="2022-01-06", params=None) -> dict:
    if source == "l1c":
        if not params:
            return get_l1c_data()
        else:
            # Add method for l1c data 
            return get_l1c_data()

    elif source == "l2a":
        if not params:
            return get_l2a_data({
                "geojson": { 
                    "type": "Feature",
                    "properties": {
                        "name": "Example name",
                        "avverkningsdatum": "2018-05-25"
                    },
                    "time": {
                        "date": date,
                    },
                    "geometry": { 
                        "type": "Box",
                        "coords": {
                            "west": 14.555719745816692,
                            "east": 14.79187736312752,
                            "south": 55.991257253340635,
                            "north": 56.10331290101734
                        }
                    },
                    "collection": "l2a",
                    "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "scl"]
                    }
                })
        else:
            return get_l2a_data(params)
    else:
        print(f"No such source as {source}")
        return None


def normalize(data:np.ndarray, _type:str = "float") -> np.ndarray:
    for i in range(data.shape[0]):
        channel = data[i, ...]
        channel_min, channel_max = np.percentile(channel, (2, 98))  # Clip values between 1st and 99th percentile
        channel = np.clip(channel, channel_min, channel_max)  # Clip the extreme values
        channel = ((channel - channel_min) / (channel_max - channel_min) * (255 if _type == "int" else 1))
        data[i, ...] = channel

    return (data.astype(np.uint8) if _type == "int" else data)

def clip(data:np.ndarray, _type:str = "float") -> np.ndarray:
    for i in range(data.shape[0]):
        channel = data[i, ...]
        channel = np.clip(channel, 0, 1)  # Clip the extreme values
        channel = channel * (255 if _type == "int" else 1) 
        data[i, ...] = channel

    return (data.astype(np.uint8) if _type == "int" else data)

def filter_array_by_bands(array:np.ndarray, array_bands: list, target_bands:list) -> np.ndarray:
    indices = [array_bands.index(value) for value in target_bands]
    return array[indices]

def plot(data, save_name: str = None) -> None:
    plt.imshow(np.transpose(data, (1, 2, 0)))

    if save_name:
        plt.savefig(save_name)

if __name__ == "__main__":

    # Bands for RGB images
    rgb_band_list = ["B04", "B03", "B02"]

    # Bands for l1c cloud-thickness-model input
    l1c_band_list = ["B02", "B03", "B04", "B05", "B06",
                    "B07", "B08", "B09", "B10", "B11", "B12"]

    # Bands for l1c cloud-thickness-model input
    l2a_band_list = ["B02", "B03", "B04", "B05", "B06",
                    "B07", "B08", "B09", "B11", "B12"]

    source = "l2a"
    data = get_data(source=source)

    RGB = get_product(data, rgb_band_list, scaling="downsizing")
    # BGR = np.asarray([data["B02"], data["B03"], data["B04"]]) # Different from RGB in plots, should not be 

    # l1c = get_product(data, l1c_band_list)

    # l2a = get_product(data, l2a_band_list)

    plot(RGB, save_name="RGB")

    plot(normalize(RGB), save_name="RGB_normalized")
    