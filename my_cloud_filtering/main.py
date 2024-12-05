import os
import time
import datetime
from shutil import copyfile
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage import measure
import json
from utils import mlp_inference, MLP5

from get_data import get_data, get_product, normalize, plot

from torch.cuda import is_available
DEVICE = "cpu"#"cuda" if is_available else "cpu"
DO_PLOT = True

MLP_POST_FILTER_SZ = 1  # 1 --> no filtering, >= 2 --> majority vote within that-sized square

# Computed at training
def mean_std_11c():
	means = torch.Tensor(np.array([0.49675034, 0.47293043, 0.564903, 0.52927473, 0.65845986, 0.93623101, 0.90515048, 0.99451205, 0.45604575, 0.07375108, 0.53309616, 0.43224668])).to(DEVICE)
	stds = torch.Tensor(np.array([0.28274442, 0.27778134, 0.28483809, 0.31573642, 0.28173209, 0.31942519, 0.32981911, 0.36159493, 0.29364748, 0.1140917,  0.41934613, 0.3335538])).to(DEVICE)
	return means, stds

# Computed at training
def mean_std_12a():
	means = torch.Tensor(np.array([0.64984976, 0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])).to(DEVICE)
	stds = torch.Tensor(np.array([0.3596485, 0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])).to(DEVICE)
	return means, stds

def init_l1c():
	input_dim = 12
	MODEL_LOAD_PATH = ['models/2023-08-10_10-33-44_model_it_2000000', 'models/2023-08-10_10-34-06_model_it_2000000', 'models/2023-08-10_10-34-18_model_it_2000000',
				   'models/2023-08-10_10-34-28_model_it_2000000', 'models/2023-08-10_10-34-46_model_it_2000000', 'models/2023-08-10_10-34-58_model_it_2000000',
				   'models/2023-08-10_10-35-09_model_it_2000000', 'models/2023-08-10_10-35-31_model_it_2000000', 'models/2023-08-10_10-35-52_model_it_2000000',
				   'models/2023-08-10_10-36-12_model_it_2000000']

	BAND_NAMES = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
	collection = "s2_msi_l1c"

	cloud_thres, thin_cloud_thres = 0.015, 0.007

	means, stds = mean_std_11c()
	return input_dim, MODEL_LOAD_PATH, BAND_NAMES, means, stds, cloud_thres, thin_cloud_thres, collection

def init_l2a():
	input_dim = 11
	MODEL_LOAD_PATH = ['models/2023-08-10_11-49-01_model_it_2000000', 'models/2023-08-10_11-49-22_model_it_2000000', 'models/2023-08-10_11-49-49_model_it_2000000',
				   'models/2023-08-10_11-50-44_model_it_2000000', 'models/2023-08-10_11-51-11_model_it_2000000', 'models/2023-08-10_11-51-36_model_it_2000000',
				   'models/2023-08-10_11-51-49_model_it_2000000', 'models/2023-08-10_11-52-02_model_it_2000000', 'models/2023-08-10_11-52-24_model_it_2000000',
				   'models/2023-08-10_11-52-47_model_it_2000000']

	BAND_NAMES = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
	collection = "s2_msi_l2a"

	cloud_thres, thin_cloud_thres = 0.02, 0.01

	means, stds = mean_std_12a()
	means = means[[1,2,3,4,5,6,7,8,9,11,12]]
	stds = stds[[1,2,3,4,5,6,7,8,9,11,12]]
	return input_dim, MODEL_LOAD_PATH, BAND_NAMES, means, stds, cloud_thres, thin_cloud_thres, collection

def init(source:str = "l1c") -> tuple:
    # init output folder
	dir = "./outputs"
	if not os.path.exists(dir):
		os.makedirs(dir)

    # get params
	if source == "l1c":
		input_dim, paths, bands, means, stds, cloud_thres, thin_cloud_thres, collection  =  init_l1c()
	else:
		input_dim, paths, bands, means, stds, cloud_thres, thin_cloud_thres, collection  = init_l2a()

	# get models
	paths = ["../"+path for path in paths]
	models = []
	for model_load_path in paths:
		model = MLP5(input_dim, 1, apply_relu=True)
		model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
		model.to(DEVICE)
		models.append(model)

	return bands, models, means, stds, cloud_thres, thin_cloud_thres, collection

def run_cloud_prediction(date: str = "2022-01-01", data_source:str = "l1c", params: dict = None):

	# Get bands, models, etc. based on the data sources
	bands, models, means, stds, cloud_thres, thin_cloud_thres, _ = init(data_source)

	# Get data
	data = get_data(source=data_source, date=date, params=params)
	if data == None:
		return

	img = get_product(data, bands, scaling="downsizing")

	rgb_band_list = ["B09", "B04", "B03", "B02"]
	RGB = normalize(get_product(data, rgb_band_list, scaling="downsizing"))[1:]

	del data

	img = np.transpose(img, (1, 2, 0))
	RGB = np.transpose(RGB, (1, 2, 0))

	# Extract image shape
	H, W = img.shape[:2]

	THRESHOLD_THICKNESS_IS_CLOUD = [cloud_thres] # 0.010  # if COT predicted above this, then predicted as 'opaque cloud' ("thick" cloud)
	THRESHOLD_THICKNESS_IS_THIN_CLOUD = [thin_cloud_thres] #0.010  # if COT predicted above this, then predicted as 'thin cloud' <-- set to the same as the opaque cloud threshold by default, i.e. it becomes a binary task (cloudy / clear) instead
	pred_map, pred_map_binary_list, pred_map_binary_thin_list = mlp_inference(img, means, stds, models, H*W,
																				THRESHOLD_THICKNESS_IS_CLOUD,
																				THRESHOLD_THICKNESS_IS_THIN_CLOUD,
																				MLP_POST_FILTER_SZ, 
																				DEVICE)

	# Track stats
	pred_map_binary = pred_map_binary_list[0]
	pred_map_binary_thin = pred_map_binary_thin_list[0]
	frac_binary = 100*np.count_nonzero(pred_map_binary + pred_map_binary_thin) / H / W
	pred_cloudy = frac_binary > 5.0

	# Visualize results     
	fig = plt.figure(figsize=(16, 16))

	fig.add_subplot(1,4,1)
	plt.imshow(RGB, vmin=0, vmax=1)
	plt.title(f'image, {"Cloudy" if pred_cloudy else "Not cloudy"}')
	plt.axis('off')

	fig.add_subplot(1,4,2)
	plt.imshow(RGB, vmin=0, vmax=1)
	if True: #SHOW_CLOUD_CONTOUR_ON_IMG:
		contours = measure.find_contours(0.0 + pred_map_binary + pred_map_binary_thin)#, 0.9)
		for contours_entry in contours:
			plt.plot(contours_entry[:, 1], contours_entry[:, 0], color='r')
	plt.title('image w contours')
	plt.axis('off')
	
	fig.add_subplot(1,4,3)
	plt.title('pred (min, max)=(%.3f, %.3f)' % (np.nanmin(pred_map), np.nanmax(pred_map)))
	pred_map[np.isnan(pred_map)] = 0
	plt.imshow(pred_map, vmin=0, vmax=1, cmap='gray')
	plt.axis('off')

	fig.add_subplot(1,4,4)
	plt.imshow(0.0 + 2*pred_map_binary + pred_map_binary_thin, vmin=0, vmax=2, cmap='gray')
	plt.title('pred-binary, cloudy (%.1f prct)' % frac_binary)
	plt.axis('off')

	print("Cloudy:", bool(pred_cloudy))
	if pred_cloudy == False:
		save_path = f"../outputs/pred_{date}_{data_source}"
		print("saving: ", save_path)
		plt.savefig(save_path)

	plt.cla()
	plt.clf()
	plt.close('all')

	print("DONE")

if __name__ == "__main__":
    
	json_coords = os.getenv("coords", default="{\"east\": 14.79187736312752, \"south\": 55.991257253340635, \"west\": 14.555719745816692, \"north\": 56.10331290101734}")
	coords = json.loads(json_coords)
	date = os.getenv("date", default="2022-01-01")
	data_source = os.getenv("data_source", default="l2a")
	print(date, data_source, coords)

	params = {
		"geojson": { 
			"time": {
				"date": date,
			},
			"geometry": { 
				"type": "Box",
    			"coords": coords
			},
			"collection": data_source,
			"bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "scl"]
			}
	}
	run_cloud_prediction(date=date, data_source=data_source, params=params)