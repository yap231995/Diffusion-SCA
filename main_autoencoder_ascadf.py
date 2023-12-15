import os
import copy
from copy import deepcopy
import random
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
from tqdm import tqdm
from src.autoencoder import Autoencoder, VAE
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.gaussian_diffusion import GaussianDiffusion1D
from src.net import Unet1D
from src.train import train, train_ae, train_vae
from src.utils import aes_label_cpa, cpa_method, unison_shuffled_copies, aes_label_cpa_ascad

dataset = "ASCAD" 
leakage = "ID"

train_ae_bool = True
cal_ae_cpa =True
print_ae_traces = True
create_dataset = True
root = './'
save_root = root + 'Result_ASCADf/' + dataset + '_' + leakage +'/'
image_root = save_root + 'image/'
latent_space_root = save_root + 'latent_space/'
new_traces_root = save_root + 'new_traces/'
if not os.path.exists(save_root):
    os.mkdir(save_root)
if not os.path.exists(image_root):
    os.mkdir(image_root)
if not os.path.exists(latent_space_root):
    os.mkdir(latent_space_root)
if not os.path.exists(new_traces_root):
    os.mkdir(new_traces_root)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloadertrain = Custom_Dataset(root = root, dataset = dataset, leakage = leakage, transform =  transforms.Compose([ ToTensor_trace() ]))
dataloadertrain.choose_phase("train_more_shares")
num_workers = 0

trace_size_original = dataloadertrain.X_profiling.shape[1]
print("original trace size: ", trace_size_original)
print("number of traces: ", dataloadertrain.X_profiling.shape[0])

rep = 100
best_corr = -1

import time
for r_xp in range(rep):
    
    # parameter to check
    dim_len_entr = random.randrange(2,6+1)       
    batch_entr = random.randrange(64,2048+1,32)  
    emb_entr = random.randrange(384,trace_size_original,32)    
    epoch_entr = random.randrange(40,100+1)     
    lr_entr = random.randrange(-5,-2+1) 

    dims = []
    for i in range((dim_len_entr)):
        dims.append(random.randrange(32,2048+1,32)) # random assignment of number of nodes

    embedding_size = copy.deepcopy(emb_entr)
    
    ae = Autoencoder(trace_size_original, embedding_size, dims).to(device)
    
    batch_size_ae = copy.deepcopy(batch_entr) # 512
    dataloaders_ae = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size_ae,
                                shuffle=True, num_workers=num_workers)}
    epochs_ae = copy.deepcopy(epoch_entr) # 50 
    lr_ae = 10**copy.deepcopy(lr_entr)# 5e-3
    
    print("dimension size:")
    print(dims)
    print("batch size, epoch size, learning rate, embedding size: ")
    print([batch_size_ae, epochs_ae, lr_ae, embedding_size])
    
    start_time = time.time()

    train_ae(dataloaders_ae, ae, epochs_ae, device, lr_ae, latent_space_root = latent_space_root)
    ae.load_state_dict(torch.load(latent_space_root + "ae_trained.pth", map_location=torch.device(device)))
    ae.eval()
    ae.cpu()
    
    with torch.no_grad():
        reconstructed_traces = ae(torch.from_numpy(dataloadertrain.X).float()).detach()
        latent_X_profiling = ae.encode(torch.from_numpy(dataloadertrain.X).float()).detach()
    latent_X_profiling = latent_X_profiling.numpy()

    trace_num_sample = copy.deepcopy(trace_size_original)
    
    if r_xp == 0:
        fig, ax = plt.subplots(figsize=(15, 7))
        x_axis = [i for i in range(trace_num_sample)]
        for i, trace in enumerate(dataloadertrain.X[:100,:]):
            trace = trace.squeeze(0)
            ax.plot(x_axis, trace)

        ax.set_xlabel('Sample points', fontsize=20)
        ax.set_ylabel('Voltage', fontsize=20)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
        plt.savefig(image_root + 'Orig_traces_' + dataset + "_" + leakage+ "_ascadf.png")

    fig, ax = plt.subplots(figsize=(15, 7))
    used_traces = 5000
    reconstructed_traces = ae(torch.from_numpy(dataloadertrain.X[:used_traces,:]).float()).detach()

    reconstructed_traces = reconstructed_traces.squeeze(1).detach()
    total_samplept = reconstructed_traces[:used_traces,:].shape[1]
    total_num_gen_trace =  reconstructed_traces[:used_traces,:].shape[0]
    
    cpa_val = np.zeros(256)
    
    if r_xp == 0:
        fig, ax = plt.subplots(figsize=(15, 7))
        real_traces = np.zeros((used_traces, dataloadertrain.X_profiling.shape[1]))
        for i, trace in enumerate(dataloadertrain.X[:used_traces,:]):
            real_traces[i,:] = trace.squeeze(0)
        
        for k in range(256):
            label_k = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces], k, leakage)
            cpa_k = cpa_method(total_samplept, total_num_gen_trace, label_k, real_traces)
        
            if dataloadertrain.correct_key == k:
                continue
            else:
                ax.plot(x_axis, cpa_k, c="grey")

        label_correct_key = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces], dataloadertrain.correct_key, leakage)
        cpa_k = cpa_method(total_samplept, total_num_gen_trace,  label_correct_key,  real_traces)
        
        x_axis = [i for i in range(total_samplept)]
        ax.plot(x_axis, cpa_k, c="red")
        plt.savefig(image_root + 'Actual_corr_' + dataset + "_" + leakage+ "_ascadf.png")

    fig, ax = plt.subplots(figsize=(15, 7))
    
    for k in range(256):
        label_k = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces], k, leakage)
        cpa_k = cpa_method(total_samplept, total_num_gen_trace, label_k, reconstructed_traces[:used_traces,:])
        
        nan_check = np.where(np.isnan(cpa_k))[0]
        if len(nan_check)==len(cpa_k):
            cpa_val[k] = np.nan
        else:
            idxx = np.where(~np.isnan(cpa_k))[0]

            cpa_val[k] = (np.max(cpa_k[idxx]))
        
        x_axis = [i for i in range(total_samplept)]
        if dataloadertrain.correct_key == k:
            continue
        else:
            ax.plot(x_axis, cpa_k, c="grey")

    label_correct_key = aes_label_cpa_ascad(dataloadertrain.plt_profiling[:used_traces], dataloadertrain.correct_key, leakage)
    cpa_k = cpa_method(total_samplept, total_num_gen_trace,  label_correct_key,  reconstructed_traces[:used_traces,:])
    
    x_axis = [i for i in range(total_samplept)]
    ax.plot(x_axis, cpa_k, c="red")
    
    nan_check = np.where(np.isnan(cpa_k))[0]
    if len(nan_check)==len(cpa_k):
        print("traces bad....")

    else:

        idxx = np.where(~np.isnan(cpa_k))[0]
        print("max correlation: ")
        print(np.max(cpa_k[idxx]))
        id_k = np.argsort([-1*p for p in cpa_val])
        print("rank correct key: ")
        print(np.where(id_k == dataloadertrain.correct_key)[0][0])
        k_rr = np.where(id_k == dataloadertrain.correct_key)[0][0]
        
        used_d = int(np.min([used_traces,5000]))
        reconstructed_traces2 = reconstructed_traces.squeeze(1)
        traces1 = dataloadertrain.X[:used_d,:]
        traces2 = reconstructed_traces2[:used_d,:]
        print("Dimension original and reconstructed traces:")
        print([traces1.shape, traces2.shape])
            
        if traces1.shape[-1] == traces2.shape[-1]:
            crr = np.zeros(used_d)
            for ii in range(used_d):
                crr[ii] = abs(np.corrcoef(traces1[ii,:], traces2[ii,:])[1, 0])
            print("average correlation between original and reconstructed traces: ")
            print(np.mean(crr))
            corr_traces = np.mean(crr)
            
        else:
            corr_traces = -1
            
        max_metric = corr_traces 
        
        if max_metric > best_corr:
            best_corr = copy.deepcopy(max_metric)
            best_cpa_corr = np.max(cpa_k[idxx])
            best_ge = np.where(id_k == dataloadertrain.correct_key)[0][0]
            best_trace_corr = copy.deepcopy(corr_traces)
            
            best_idx = copy.deepcopy(r_xp)
            best_dim = copy.deepcopy(dims)
            best_param = [batch_size_ae, epochs_ae, lr_ae, embedding_size]
            plt.savefig(image_root + 'Corr_' + dataset + "_" + leakage+ "_ascadf.png")
            
            trace_num_sample = copy.deepcopy(trace_size_original)
            fig, ax = plt.subplots(figsize=(15, 7))
            x_axis = [i for i in range(trace_num_sample)]

            for i, trace in enumerate(reconstructed_traces2[:100,:]):
                ax.plot(x_axis, trace)

            ax.set_xlabel('Sample points', fontsize=20)
            ax.set_ylabel('Voltage', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)

            plt.savefig(image_root + 'Gen_traces_' + dataset + "_" + leakage+ "_ascadf.png")
            
            path_dataset = latent_space_root + "latent_dataset" + "/"
            if not os.path.exists(path_dataset):
                os.mkdir(path_dataset)
            with torch.no_grad():
                latent_X_profiling = ae.encode(torch.from_numpy(dataloadertrain.X_profiling).float()).detach()
                latent_X_attack = ae.encode(torch.from_numpy(dataloadertrain.X_attack).float()).detach()
            np.save(path_dataset + "X_latent_profiling.npy", latent_X_profiling)
            np.save(path_dataset + "X_latent_attack.npy", latent_X_attack)
            np.save(path_dataset + "Y_profiling.npy", dataloadertrain.Y_profiling)
            np.save(path_dataset + "Y_attack.npy", dataloadertrain.Y_attack)
            np.save(path_dataset + "plt_profiling.npy", dataloadertrain.plt_profiling)
            np.save(path_dataset + "plt_attack.npy", dataloadertrain.plt_attack)
            np.save(path_dataset + "correct_key.npy", dataloadertrain.correct_key)
            np.save(path_dataset + "gen_traces.npy", reconstructed_traces)
            torch.save(ae.state_dict(), path_dataset + "ae_trained.pth")
            
            best_params_tmp = dict(dims=best_dim, batch_size=best_param[0], epoch_size=best_param[1], learn_rate=best_param[2], embedding_size=best_param[3], max_cpa=best_corr, key_rank=best_ge)

            import pickle
            with open(path_dataset + 'params_tmp_' + dataset + "_" + leakage+ '.pkl', 'wb') as fp:
                pickle.dump(best_params_tmp, fp)
                print('Your dictionary has been saved successfully to file')
            
            used_d = int(np.min([used_traces,5000]))
            traces1 = dataloadertrain.X[:used_d,:]
            traces2 = reconstructed_traces2[:used_d,:]
            print("Dimension original and reconstructed traces:")
            print([traces1.shape, traces2.shape])
            
            if traces1.shape[-1] == traces2.shape[-1]:
                crr = np.zeros(used_d)
                for ii in range(used_d):
                    crr[ii] = abs(np.corrcoef(traces1[ii,:], traces2[ii,:])[1, 0])
                print("average correlation original and reconstructed traces: ")
                print(np.mean(crr))

    
    print([r_xp, time.time()-start_time])
    print("")
    plt.clf()
    plt.cla()
    plt.close()
    
print("Experiment Done...\n")
print("Best parameters achieved at iteration {:d} ".format(int(best_idx)))
print("Best dimension: [ ", end = '')
for i in range(len(best_dim)):
    print("{:d} ".format(int(best_dim[i])), end = '')
print("]")
print("Best batch size: {:d} ".format(int(best_param[0])))
print("Best epoch size: {:d} ".format(int(best_param[1])))
print("Best learning rate: {:.11f} ".format(best_param[2]))
print("Best embedding size: {:d} ".format(int(best_param[3])))
print("Max (abs) CPA: {:.2f} ".format(best_cpa_corr))
print("Best GE: {:d} ".format(best_ge))
print("Max (abs) traces correlation: {:.2f} ".format(best_trace_corr))

best_params = dict(dims=best_dim, batch_size=best_param[0], epoch_size=best_param[1], learn_rate=best_param[2], embedding_size=best_param[3], max_cpa=best_corr, key_rank=best_ge)

import pickle
with open(path_dataset + 'params_' + dataset + "_" + leakage+ '.pkl', 'wb') as fp:
    pickle.dump(best_params, fp)
    print('Your dictionary has been saved successfully to file')
    
with open(path_dataset + 'params_' + dataset + "_" + leakage+ '.pkl', 'rb') as fp:
    std = pickle.load(fp)
    print('the dict:')
    print(std)