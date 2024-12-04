# from utils import MLP5
# def l1c_models_paths():
#     ['../log/2023-08-10_10-33-44/model_it_2000000', '../log/2023-08-10_10-34-06/model_it_2000000', '../log/2023-08-10_10-34-18/model_it_2000000',
# 				   '../log/2023-08-10_10-34-28/model_it_2000000', '../log/2023-08-10_10-34-46/model_it_2000000', '../log/2023-08-10_10-34-58/model_it_2000000',
# 				   '../log/2023-08-10_10-35-09/model_it_2000000', '../log/2023-08-10_10-35-31/model_it_2000000', '../log/2023-08-10_10-35-52/model_it_2000000',
# 				   '../log/2023-08-10_10-36-12/model_it_2000000']
    
# def l2a_models_paths():
#     return ['../log/2023-08-10_11-49-01/model_it_2000000', '../log/2023-08-10_11-49-22/model_it_2000000', '../log/2023-08-10_11-49-49/model_it_2000000',
# 				   '../log/2023-08-10_11-50-44/model_it_2000000', '../log/2023-08-10_11-51-11/model_it_2000000', '../log/2023-08-10_11-51-36/model_it_2000000',
# 				   '../log/2023-08-10_11-51-49/model_it_2000000', '../log/2023-08-10_11-52-02/model_it_2000000', '../log/2023-08-10_11-52-24/model_it_2000000',
# 				   '../log/2023-08-10_11-52-47/model_it_2000000']

# def get_models(model_paths):
#     # Setup and load model
# 	models = []
# 	for model_load_path in model_paths:
# 		model = MLP5(input_dim, output_dim, apply_relu=True)
# 		model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
# 		model.to(DEVICE)
# 		models.append(model)

# def get_model(data_type: str = "l2a"):
#     if data_type == "l2a":
#        	return get_models(l2a_models_paths())
# 	elif data_type == "l1c":
#         return get_models(l1c_models_paths())