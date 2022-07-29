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
args.obs_len = 8
args.pred_len = 8
args.model_type = "spatial_temporal"
args.dset_name = "zara1"
args.best_k = 5
args.l = 0.1
args.delim = "\t"

def plot_pedestrian(seq, pred, c):
	# Number of trajectories x prediction length x 2 
	# Plot Ground Truth
	sns.lineplot(seq[...,0], seq[...,1],marker='o', markersize=8, linewidth=1.0, color='red')
	traj = torch.cat((seq[...,-1,:].unsqueeze(0).repeat(pred.shape[0], 1, 1), pred), dim=1)
	# Plot Predictions
	for i in range(traj.shape[0]):
		sns.lineplot(traj[i,...,0], traj[i,...,1], marker='o', markersize=8, color='blue', linestyle='-', linewidth=0.8, zorder=1)

def get_prediction(batch, model, args, generative=False):
	# Generate Predictions
	batch = get_batch(batch)	
	predictions, _, sequence, _, _, _  = predict(batch, model)
	return predictions.unsqueeze(1), sequence

def reload_data():
    testdataset = dataset(glob.glob(f'data/MATRiX/sample.txt'), args)
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

dirname = f'plots/{k}-{l}-{args.dset_name}' if generative else f'plots/MATRiX'
if not os.path.exists(dirname): os.makedirs(dirname)

batchNum = 0
img_array = []

def MATRIX_predictions(array: [], num: int):
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
            pred_p1 = predictions[:,p1,...]
            #print(pred_p1)

        colors = plt.cm.tab10(np.linspace(0,1,num_ped))
        print(f'Number of pedestrians: {num_ped}')
        fig, ax = plt.subplots()
        #img = plt.imread(f'zara01/img-{b+1}.png') # Background Image
        #plt.imshow(img, alpha=0.6, extent = [0, 16, 0, 14], zorder=0)
        for p1 in range(num_ped):
            seq_p1 = sequence[p1,...]
            pred_p1 = predictions[:,p1,...]
            plot_pedestrian(seq_p1, pred_p1, colors[p1])
        
        legend_elements = [Line2D([0], [0], marker='o', color='r', label='Ground Truth',
                            markerfacecolor='r', markersize=8), Line2D([0], [0], marker='o', color='blue', label='Prediction',
                            markerfacecolor='blue', markersize=8)]
        ax.legend(handles=legend_elements, loc=(1.05, 0))
        plt.title(args.dset_name.upper())
        plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.tight_layout()
        plt.xlim([0, 16])
        
        plt.ylim([0, 14])
        plt.savefig(f'{dirname}/{num}.png')
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        array+=[data]

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
            MATRIX_predictions(img_array, batchNum)
            batchNum += 1
            #conn.sendall("".encode())