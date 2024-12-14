import json 
import os
import math

# Sweden is 450,295 km^2
# Masking sweden by 1km x 1km at a time, with xmin for each call,
#     we need x workers to do this in under 24h, each day.

# Based on the Sölvesborg image, approx 12x14km,
# 1 workers: 61 hours, 44 minutes, 24 seconds
# 2 workers: 30 hours, 53 minutes, 42 seconds
# 4 workers: 15 hours, 26 minutes, 51 seconds
# 8 workers: 7 hours, 43 minutes, 5 seconds
# 12 workers: 5 hours, 9 minutes, 20 seconds
# 20 workers: 3 hours, 5 minutes, 12 seconds

def latitude_to_meters(latitude):
    40075/360*math.cos(latitude) # In km

def partition_coords():
    return

def run_large_cloudmask(date, data_source, params):
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
 
    # def get_cloud mask():
    # SPLIT PRED MAP INTO PEICES AND ASSERT CLOUD MASK FOR EACH PART
    #     return None

    # Track stats
    pred_map_binary = pred_map_binary_list[0]
    pred_map_binary_thin = pred_map_binary_thin_list[0]
    frac_binary = 100*np.count_nonzero(pred_map_binary + pred_map_binary_thin) / H / W
    pred_cloudy = frac_binary > 5.0

    print("Cloudy:", bool(pred_cloudy))
    if pred_cloudy == False:
		# DO SOMETHING, STORE THE BOOLs AS ARRAY OR SIMILAR

if __name__ == "__main__":

    json_coords = os.getenv("coords", default=None)
    if not json_coords:
        coords = json.loads(json_coords)
    else:
        from max_coords import *
        coords = {}
        coords["north"] = max_north
        coords["south"] = max_south
        coords["east"] = max_east
        coords["west"] = max_west

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