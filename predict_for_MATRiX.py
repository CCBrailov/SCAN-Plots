import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
#from torch.autograd import Variable

import glob

import matplotlib
import seaborn as sns
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from torch.utils.data import DataLoader
from arguments import parse_arguments
from model import TrajectoryGenerator
from data import dataset, collate_function

from generative_utils import *
from utils import *

import socket

HOST = "127.0.0.1"
PORT = 2063

# Parse Arguments
args = parse_arguments()
args.model_type = "spatial_temporal"
args.dset_name = "zara2"
args.best_k = 5
args.l = 0.1
args.delim = "\t"

def get_prediction(batch, model, args, generative=False):
	# Generate Predictions
	batch = get_batch(batch)	
	predictions, _, sequence, _, _, _  = predict(batch, model)
	return predictions.unsqueeze(1), sequence

def reload_data():
    testdataset = dataset(glob.glob(f'data/zara1/test/temp.txt'), args)
    print(f'Number of Test Samples: {len(testdataset)}')
    print('-'*100)
    return testdataset

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Test Dataset
#testdataset = reload_data()

# Initialize DataLoader
#testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
k = args.best_k
l = args.l

# Initialize Model, Load Saved Weights
generative = ('generative' in args.model_type)
model = TrajectoryGenerator(model_type=args.model_type, obs_len=args.obs_len, pred_len=args.pred_len, feature_dim=2, embedding_dim=args.embedding_dim, encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim, domain_parameter=args.domain_parameter, delta_bearing=args.delta_bearing, delta_heading=args.delta_heading, pretrained_scene='resnet18', device=device, noise_dim=args.noise_dim if generative else None, noise_type=args.noise_type if generative else None).float().to(device)

if generative:
	model_file = f'./trained-models/{args.model_type}/{args.dset_name}/{args.best_k}V-{args.l}_g.pt'
	
else:
	model_file = f'./trained-models/{args.model_type}/{args.dset_name}.pt'

model.load_state_dict(torch.load(model_file))

def MATRIX_predictions():
    testdataset = reload_data()
    testloader = DataLoader(testdataset, batch_size=1, collate_fn=collate_function(), shuffle=False)
    for b, batch in enumerate(testloader):
        sequence, target, dist_matrix, bearing_matrix, heading_matrix, ip_mask, \
        op_mask, pedestrians, batch_mean, batch_var = batch
        if pedestrians.data<2:
            continue
        predictions, sequence = get_prediction(batch, model, args)	
        predictions = predictions.squeeze(0)
        predictions = predictions.clone().detach().cpu()
        sequence = sequence.squeeze(0).clone().detach().cpu()
        target = target.squeeze(0).clone().detach().cpu()
        gt_traj = torch.cat((sequence, target), dim=1)
        num_ped, slen = sequence.size()[:2]

        print(f"Predicting trajectories for {num_ped} agents")

        #Empty array to hold prediction strings
        pStrings = []

        #For each pedestrian in the scene
        for p1 in range(num_ped):
            seq_p1 = sequence[p1,...]
            #Retrieve predictions
            pred_p1 = torch.Tensor.tolist(predictions[:,p1,...])[0]
            #Create empty prediction string
            pString = ""
            #For each x/y pair, add {x}/{y} to the string followed by a comma
            for pair in pred_p1:
                pString += f"{pair[0]}/{pair[1]}"
                pString += ","
            pString = pString[:-1]            #Remove the last comma
            pStrings.append(pString)
        
        #Empty string to eventually pass through server
        dataString = ""

        for string in pStrings:
            dataString += f"{string}|"
        
        dataString = dataString[:-1]
        return dataString

# Start server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            data = data.decode()
            # Do whatever with data
            dataString = MATRIX_predictions()
            conn.sendall(dataString.encode())