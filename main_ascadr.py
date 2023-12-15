import os
from copy import deepcopy
import random
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
from tqdm import tqdm

from src.autoencoder import Autoencoder
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.gaussian_diffusion import GaussianDiffusion1D
from src.net import Unet1D, Unet1D_more_shares
from src.train import train, train_ae
from src.utils import aes_label_cpa, cpa_method, unison_shuffled_copies, aes_label_cpa_mask, multiply_sample_pt, \
    obtain_plaintext, aes_label_cpa_ascad

dataset = "ASCAD_variable" 
leakage = "ID"

dataset2 = dataset + "_AE"

batch_size = 200
epochs = 390
lr = 0.005

print_orig_cpa = True
training_diffusion = True
sampling = True
print_traces = True
cal_cpa = True

root = './'
save_root = root + 'Result_ASCADr/' + dataset + '_' + leakage +'/'
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
    
save_root2 = root + 'Result_ASCADr/' + dataset2 + '_' + leakage +'/'
image_root2 = save_root2 + 'image/'
latent_space_root2 = save_root2 + 'latent_space/'
new_traces_root2 = save_root2 + 'new_traces/'
if not os.path.exists(save_root2):
    os.mkdir(save_root2)
if not os.path.exists(image_root2):
    os.mkdir(image_root2)
if not os.path.exists(latent_space_root2):
    os.mkdir(latent_space_root2)
if not os.path.exists(new_traces_root2):
    os.mkdir(new_traces_root2)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path_dataset = latent_space_root + "latent_dataset" + "/"
latent_X_profiling=np.load(path_dataset + "X_latent_profiling.npy")
latent_X_attack = np.load(path_dataset + "X_latent_attack.npy")
plt_profiling = np.load(path_dataset + "plt_profiling.npy")
plt_attack = np.load(path_dataset + "plt_attack.npy")
label_profiling = np.load(path_dataset + "Y_profiling.npy")
label_attack = np.load(path_dataset + "Y_attack.npy")
correct_key = np.load(path_dataset + "correct_key.npy")

dataloadertrain_X = Custom_Dataset(root = root, dataset = dataset, leakage = leakage, transform =  transforms.Compose([ ToTensor_trace() ]))

fig_gen, ax_gen = plt.subplots(figsize=(15, 7))
x_axis = [i for i in range(dataloadertrain_X.X_profiling.shape[1])]
fig_old_latent, ax_old_latent = plt.subplots(figsize=(15, 7))
for i, trace in enumerate(dataloadertrain_X.X_profiling[:100,:]):

    ax_old_latent.plot(x_axis, trace)

ax_old_latent.set_xlabel('Sample points', fontsize=20)
ax_old_latent.set_ylabel('Voltage', fontsize=20)
        
for label in (ax_gen.get_xticklabels() + ax_gen.get_yticklabels()):
    label.set_fontsize(15)
plt.savefig(image_root2 + 'original_traces_' + dataset2 + "_" + leakage+ "_norm.png")

mean_trace = np.mean(dataloadertrain_X.X_profiling, axis=0)
std_trace = np.std(dataloadertrain_X.X_profiling, axis=0)

fig_gen, ax_gen = plt.subplots(figsize=(15, 7))
x_axis = [i for i in range(latent_X_profiling.shape[1])]
fig_old_latent, ax_old_latent = plt.subplots(figsize=(15, 7))
for i, trace in enumerate(latent_X_profiling[:100,:]):

    ax_old_latent.plot(x_axis, trace)

ax_old_latent.set_xlabel('Sample points', fontsize=20)
ax_old_latent.set_ylabel('Voltage', fontsize=20)
        
for label in (ax_gen.get_xticklabels() + ax_gen.get_yticklabels()):
    label.set_fontsize(15)
    # plt.show()
plt.savefig(image_root2 + 'Latent_original_traces_' + dataset2 + "_" + leakage+ "_norm.png")

mean_latent = np.mean(latent_X_profiling, axis=0)
std_latent = np.std(latent_X_profiling, axis=0)

import pickle 

with open(path_dataset + 'params_ascad_variable.pkl', 'rb') as fp:
    std = pickle.load(fp)
    print('the parameter:')
    print(std)

embedding_size = latent_X_profiling.shape[1]
if dataset == "AES_HD_ext":
    trace_size_original = 1264 # after pad
elif dataset == "Chipwhisperer":
    trace_size_original = 5000
elif dataset == "ASCAD":
    trace_size_original = 704 # after pad
elif dataset == "ASCAD_variable":
    trace_size_original = 1404 # after pad
dims = std['dims'][:-1]
print("dimensions")
print(dims)
print("no of traces: ", latent_X_profiling.shape[0])
decoder = True#False
if dataset == "latent_simulated_traces_order_0":
    embedding_size = 24
    trace_size_original = 100
    decoder = True
    dims = [50]

dataloadertrain = Custom_Dataset(root = root, dataset = dataset2, leakage = leakage,embedding_size = embedding_size, AE_path  = path_dataset, transform =  transforms.Compose([ ToTensor_trace() ]))

dataloadertrain.choose_phase("train_more_shares")

num_workers = 0

trace_size = dataloadertrain.X_profiling.shape[1]
print("trace_size: ", trace_size)
dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers),}

if leakage == "HW":
    classes = 9
elif leakage == "ID":
    classes = 256
if dataset == "simulated_traces_order_0" or dataset == "Chipwhisperer":
    masking_order = 0
elif dataset == "simulated_traces_order_1" or dataset == "AES_HD_ext_plaintext" or dataset == "AES_HD_ext_sbox" or dataset == "AES_HD_ext_label" or dataset == "AES_HD_ext" or dataset == "ASCAD" or dataset == "ASCAD_variable":
    masking_order = 1
elif dataset == "simulated_traces_order_2":
    masking_order = 2
elif dataset == "simulated_traces_order_3":
    masking_order = 3
model_path = save_root+dataset+"_"+leakage+"_epochs_"+str(epochs) +"_more_shares"

timestamp = 4000
print("timestamp:", timestamp)

model = Unet1D_more_shares(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1,
    masking_order = masking_order,
    num_classes=classes
).to(device)
ema_model = Unet1D_more_shares(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1,
    masking_order = masking_order,
    num_classes=classes
).to(device)
diffusion = GaussianDiffusion1D(
    model,
    device,
    seq_length = trace_size,
    timesteps = timestamp,
    objective = 'pred_noise'
)

if training_diffusion == True:
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%Hh%Mm%Ss_")
    tensorboard_root = save_root + 'tensorboard_log' + dt_string +'/'
    if not os.path.exists(tensorboard_root):
        os.mkdir(tensorboard_root)
    ema_model,ema = train(dataloaders,diffusion,device,lr, epochs,dataset, save_model_root=model_path)
else:
    ema_model.load_state_dict(torch.load(model_path+"_ema.pth", map_location=torch.device(device)))
    model.load_state_dict(torch.load(model_path+"_original.pth", map_location=torch.device(device)))

with torch.no_grad():
    diffusion_ema = GaussianDiffusion1D(
        ema_model.eval(),
        device,
        seq_length = trace_size,
        timesteps = timestamp,
        objective = 'pred_noise'
    )

    new_traces = []
    new_masks = []
    for class_index in range(0, classes):
        if "simulated" in dataset:
            batch_size = 20 
        else:
            batch_size = 10
        new_plaintext_class = class_index * torch.ones((batch_size,), dtype=torch.int).to(device)
        new_masks.append(new_plaintext_class.cpu().numpy())

    new_masks = np.concatenate(new_masks, axis=0)
    if masking_order == 0:
        new_masks = np.expand_dims(new_masks, axis=1)
        print("new_masks: ", new_masks.shape)
        print("new_masks: ", new_masks)

    elif masking_order == 1:
        print("MASKING ORDER 1")
        new_masks = np.expand_dims(new_masks, axis=1)
        masking1 =np.random.randint(0,255, size = new_masks.shape)
        new_masks = np.concatenate([new_masks, masking1], axis=1)  # This should include all masking.
        print("new_masks: ", new_masks.shape)
    elif masking_order == 2:
        new_masks = np.expand_dims(new_masks, axis=1)
        masking1 =np.random.randint(0,255, size = new_masks.shape)
        masking2 =np.random.randint(0,255, size = new_masks.shape)
        print("new_masks: ", new_masks.shape)
        print("masking1: ", masking1.shape)
        print("masking2: ", masking2.shape)

        new_masks = np.concatenate((new_masks, masking1, masking2), axis = 1) #This should include all masking.
        print("new_masks: ", new_masks.shape)
    elif masking_order == 3:
        new_masks = np.expand_dims(new_masks, axis=1)
        masking1 =np.random.randint(0,255, size = new_masks.shape)
        masking2 =np.random.randint(0,255, size = new_masks.shape)
        masking3 =np.random.randint(0,255, size = new_masks.shape)
        print("new_masks: ", new_masks.shape)
        print("masking1: ", masking1.shape)
        print("masking2: ", masking2.shape)
        print("masking3: ", masking2.shape)

        new_masks = np.concatenate((new_masks, masking1, masking2, masking3), axis = 1) 
        print("new_masks: ", new_masks.shape)

    if sampling == True:
        clip_denoised= False
        if "simulated" in dataset:
            new_latent_traces= diffusion_ema.sample(torch.from_numpy(new_masks).to(device), batch_size = batch_size*classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised= clip_denoised)
            new_latent_traces = new_latent_traces.cpu().numpy()
        else:
            collect_all_latent_traces = np.zeros((new_masks.shape[0], 1, dataloadertrain.X_profiling.shape[1]))

            step = 50
            print("new_plaintext.shape: ", new_masks.shape)
            num_new_plaintext = 0
            while num_new_plaintext <= (new_masks.shape[0])-1:
                print("num_new_plaintext: ", num_new_plaintext)
                print("num_new_plaintext : ", num_new_plaintext )
                print("num_new_plaintext + step: ", num_new_plaintext + step)
                sampling_from_ptx = new_masks[num_new_plaintext: num_new_plaintext + step]
                print("sampling_from_ptx.shape: ", sampling_from_ptx.shape)
                num_sample_ptx = sampling_from_ptx.shape[0]
                new_latent_traces = diffusion_ema.sample(torch.from_numpy(sampling_from_ptx).to(device),
                                                         batch_size=num_sample_ptx, cond_scale=6., rescaled_phi=0.7,
                                                         clip_denoised=clip_denoised)

                collect_all_latent_traces[num_new_plaintext : num_new_plaintext + step] = new_latent_traces.cpu().numpy()

                num_new_plaintext += step
            new_latent_traces = collect_all_latent_traces
            
            for x2 in range(new_latent_traces.shape[2]):

                mean_latent_new = np.mean(new_latent_traces[:,0,x2])
                std_latent_new = np.std(new_latent_traces[:,0,x2])
                a_shift = std_latent[x2]/std_latent_new
                b_shift =  mean_latent[x2] - a_shift* mean_latent_new

                for x1 in range(new_latent_traces.shape[0]):
                    new_latent_traces[x1,0,x2] = a_shift * new_latent_traces[x1,0,x2] + b_shift

            x_mean = np.zeros(new_latent_traces.shape[2])
            x_std = np.zeros(new_latent_traces.shape[2])

            for x1 in range(new_latent_traces.shape[2]):
                x_mean[x1] = np.mean(new_latent_traces[:,0,x1])
                x_std[x1] = np.std(new_latent_traces[:,0,x1])
           
        np.save(new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_more_shares.npy",
                new_latent_traces)
        np.save(new_traces_root + "diffusion_labels_masks_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_more_shares.npy",
                new_masks)
    else:
        new_latent_traces = np.load(new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_more_shares.npy",)
        new_masks = np.load(new_traces_root + "diffusion_labels_masks_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_more_shares.npy" )
        
    print("new_traces: ", new_latent_traces.shape)

    if decoder == False:
        new_traces = new_latent_traces
    elif decoder == True:
        save_ae_new_traces = True
        if save_ae_new_traces == True:
            ae = Autoencoder(trace_size_original, embedding_size, dims)
            ae.load_state_dict(torch.load(save_root.replace("latent_", "")+"latent_space/latent_dataset/" + "ae_trained.pth", map_location=torch.device("cpu")))
            new_traces = ae.decode(torch.from_numpy(new_latent_traces).float()).detach()
            new_traces = new_traces.cpu().numpy()

            for x2 in range(new_traces.shape[2]):

                mean_new = np.mean(new_traces[:,0,x2])
                std_new = np.std(new_traces[:,0,x2])
                if std_new>0:
                    a_shift = std_trace[x2]/std_new
                else:
                    a_shift = 1
                b_shift =  mean_trace[x2] - a_shift* mean_new

                for x1 in range(new_traces.shape[0]):
                    new_traces[x1,0,x2] = a_shift * new_traces[x1,0,x2] + b_shift

            x_mean = np.zeros(new_traces.shape[2])
            x_std = np.zeros(new_traces.shape[2])
            for x1 in range(new_traces.shape[2]):
                x_mean[x1] = np.mean(new_traces[:,0,x1])
                x_std[x1] = np.std(new_traces[:,0,x1])
            
            np.save(new_traces_root + "diffusion_new_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_more_shares.npy",new_traces)
        else:
            new_traces = np.load(new_traces_root + "diffusion_new_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_more_shares.npy" )
            

if print_traces == True:
    fig_gen, ax_gen = plt.subplots(figsize=(15, 7))
    new_traces = new_traces.squeeze(1)
    x_axis = [i for i in range( new_traces.shape[1])]

    for i, trace in enumerate(new_traces[:300,:]):
        ax_gen.plot(x_axis, trace)

    ax_gen.set_xlabel('Sample points', fontsize=20)
    ax_gen.set_ylabel('Voltage', fontsize=20)
    for label in (ax_gen.get_xticklabels() + ax_gen.get_yticklabels()):
        label.set_fontsize(15)
    plt.savefig(image_root + 'Generated_traces_' + dataset + "_" + leakage+"_epochs_"+str(epochs) + "_batch_" + str(batch_size) + "_more_shares.png")

    if decoder == True:
        trace_num_sample_latent = embedding_size
        fig_new_latent, ax_new_latent = plt.subplots(figsize=(15, 7))
        x_axis = [i for i in range(trace_num_sample_latent)]
        new_latent_traces = new_latent_traces.squeeze(1)

        for i, trace in enumerate(new_latent_traces[:100,:]):

            ax_new_latent.plot(x_axis, trace)

        ax_new_latent.set_xlabel('Sample points', fontsize=20)
        ax_new_latent.set_ylabel('Voltage', fontsize=20)
        for label in (ax_gen.get_xticklabels() + ax_gen.get_yticklabels()):
            label.set_fontsize(15)

        plt.savefig(image_root + 'Latent_generated_traces_' + dataset + "_" + leakage + "_epochs_" + str(epochs) + "_batch_" + str(batch_size) + "_more_shares.png")
    plt.cla()

if cal_cpa == True:
    fig, ax = plt.subplots(figsize=(15, 7))
    print(new_traces.shape)
    if masking_order == 0:
        label_name = "$Sbox(pt\oplus k^*)$"
    elif masking_order == 1:
        label_name = "$Sbox(pt\oplus k^*) \oplus m_1$"
    elif masking_order == 2:
        label_name = "$Sbox(pt\oplus k^*) \oplus m_1 \oplus m2$"
    elif masking_order == 3:
        label_name = "$Sbox(pt\oplus k^*) \oplus m_1 \oplus m_2 \oplus m_3$"
    if print_traces == False:
        new_traces = new_traces.squeeze(1)
    
    total_samplept = new_traces.shape[1]
    total_num_gen_trace = new_traces.shape[0]

    
    if dataset == "AES_HD_ext_plaintext" or dataset == "AES_HD_ext_sbox" or dataset == "AES_HD_ext_label" or dataset == "AES_HD_ext":
        pass
    else:
        print("dataset: ", dataset)
        
        print("mask_shares size: ", new_masks.shape)

        for shares in range(masking_order+1):

            tmmp = new_masks[:, shares]

            cpa_k_m = cpa_method(total_samplept, total_num_gen_trace, new_masks[:, shares], new_traces)
            x_axis = [i for i in range(total_samplept)]
            if shares == 0:

                ax.plot(x_axis, cpa_k_m, c="red", label = "$Sbox(pt\oplus k) \oplus r$")
            elif shares == 1:

                ax.plot(x_axis, cpa_k_m, c="blue", label = "$r$")
            elif shares == 2:
                ax.plot(x_axis, cpa_k_m, c="orange", label = "$m_3$")
    ax.set_xlabel('Sample points', fontsize=20)
    ax.set_ylabel('(Absolute) Correlation', fontsize=20)
    for label_fig in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_fig.set_fontsize(20)

    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
    plt.savefig(image_root + 'CPA_generated_' + dataset + "_" + leakage + "_epochs_" + str(epochs) + "_more_shares.png")
    plt.show()









