import os
from copy import deepcopy
import random
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.autoencoder import Autoencoder
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.gaussian_diffusion import GaussianDiffusion1D
from src.net import Unet1D, Unet1D_more_shares
from src.train import train, train_ae
from src.utils import aes_label_cpa, cpa_method, unison_shuffled_copies, aes_label_cpa_mask, multiply_sample_pt, \
    obtain_plaintext, scale_transform_trace

dataset = "simulated_traces_order_3"  # simulated_traces_order_3 #Chipwhisperer #AES_HD_ext_plaintext #AES_HD_ext_sbox #AES_HD_ext_label #latent_simulated_traces_order_0
leakage = "ID"
batch_size = 512  # simulated_data: 512,  CW: 50
epochs = 50  # simulated_data: 50,  CW: 100
lr = 0.0005

training_diffusion = False
sampling = False
print_traces = True
cal_cpa = True
root = './'
save_root = root + 'Result/' + dataset + '_' + leakage + '/'
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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

embedding_size = None
decoder = False
if dataset == "latent_simulated_traces_order_0":
    embedding_size = 24
    trace_size_original = 100
    decoder = True
    dims = [50]

dataloadertrain = Custom_Dataset(root=root, dataset=dataset, leakage=leakage, embedding_size=embedding_size,
                                 transform=transforms.Compose([ToTensor_trace()]))
# dataloaderval = deepcopy(dataloadertrain)
if "latent" not in dataset:
    dataloadertrain.apply_MinMaxScaler()

dataloadertrain.choose_phase("train_more_shares")

num_workers = 0

trace_size = dataloadertrain.X_profiling.shape[1]
print("trace_size: ", trace_size)
print("number of traces: ", dataloadertrain.X_profiling.shape[0])
dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                    shuffle=True, num_workers=num_workers), }
# "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
#                               shuffle=True, num_workers=num_workers)
# print(ok)
if leakage == "HW":
    classes = 9
elif leakage == "ID":
    classes = 256
if dataset == "simulated_traces_order_0" or dataset == "Chipwhisperer":
    masking_order = 0
elif dataset == "simulated_traces_order_1" or dataset == "AES_HD_ext_plaintext" or dataset == "AES_HD_ext_sbox" or dataset == "AES_HD_ext_label":  # note, aes_hd_ext is using two plaintext instead of shares.
    masking_order = 1
elif dataset == "simulated_traces_order_2":
    masking_order = 2
elif dataset == "simulated_traces_order_3":
    masking_order = 3
model_path = save_root + dataset + "_" + leakage + "_epochs_" + str(epochs) + "_more_shares"

timestamp = 4000
print("timestamp:", timestamp)

model = Unet1D_more_shares(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,
    masking_order=masking_order,
    num_classes=classes
).to(device)
ema_model = Unet1D_more_shares(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,
    masking_order=masking_order,
    num_classes=classes
).to(device)
diffusion = GaussianDiffusion1D(
    model,
    device,
    seq_length=trace_size,
    timesteps=timestamp,
    # sampling_timesteps= 500,
    objective='pred_noise'
)

print(model)
if training_diffusion == True:
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%Hh%Mm%Ss_")
    tensorboard_root = save_root + 'tensorboard_log' + dt_string + '/'
    if not os.path.exists(tensorboard_root):
        os.mkdir(tensorboard_root)
    writer = SummaryWriter(tensorboard_root)
    ema_model, ema = train(dataloaders, diffusion, device, lr, epochs, writer, dataset, save_model_root=model_path)
    writer.flush()
    writer.close()
else:
    ema_model.load_state_dict(torch.load(model_path + "_ema.pth", map_location=torch.device(device)))
    model.load_state_dict(torch.load(model_path + "_original.pth", map_location=torch.device(device)))
# print(ema_model)
with torch.no_grad():
    diffusion_ema = GaussianDiffusion1D(
        ema_model.eval(),
        device,
        seq_length=trace_size,
        timesteps=timestamp,
        # sampling_timesteps= 500,
        objective='pred_noise'
    )

    new_traces = []
    new_masks = []
    # Create the labels and the shares.
    for class_index in range(0, classes):
        # print("class_index: ", class_index)
        if "simulated" in dataset:
            batch_size = 10  # simulated data: 10, CW: 128
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
        new_masks = np.expand_dims(new_masks, axis=1)
        masking1 = np.random.randint(0, 255, size=new_masks.shape)
        new_masks = np.concatenate([new_masks, masking1], axis=1)  # This should include all masking.
        print("new_masks: ", new_masks.shape)
    elif masking_order == 2:
        new_masks = np.expand_dims(new_masks, axis=1)
        masking1 = np.random.randint(0, 255, size=new_masks.shape)
        masking2 = np.random.randint(0, 255, size=new_masks.shape)
        print("new_masks: ", new_masks.shape)
        print("masking1: ", masking1.shape)
        print("masking2: ", masking2.shape)

        new_masks = np.concatenate((new_masks, masking1, masking2), axis=1)  # This should include all masking.
        print("new_masks: ", new_masks.shape)
    elif masking_order == 3:
        new_masks = np.expand_dims(new_masks, axis=1)
        masking1 = np.random.randint(0, 255, size=new_masks.shape)
        masking2 = np.random.randint(0, 255, size=new_masks.shape)
        masking3 = np.random.randint(0, 255, size=new_masks.shape)
        print("new_masks: ", new_masks.shape)
        print("masking1: ", masking1.shape)
        print("masking2: ", masking2.shape)
        print("masking3: ", masking2.shape)

        new_masks = np.concatenate((new_masks, masking1, masking2, masking3),
                                   axis=1)  # This should include all masking.
        print("new_masks: ", new_masks.shape)
    # Starts sampling.
    if sampling == True:
        clip_denoised = False
        if "simulated" in dataset:
            new_latent_traces = diffusion_ema.sample(torch.from_numpy(new_masks).to(device),
                                                     batch_size=batch_size * classes, cond_scale=6., rescaled_phi=0.7,
                                                     clip_denoised=clip_denoised)
            new_latent_traces = new_latent_traces.cpu().numpy()
        else:
            collect_all_latent_traces = np.zeros((new_masks.shape[0], 1, dataloadertrain.X_profiling.shape[1]))

            step = 50
            print("new_plaintext.shape: ", new_masks.shape)
            num_new_plaintext = 0
            while num_new_plaintext <= (new_masks.shape[0]):
                print("num_new_plaintext: ", num_new_plaintext)
                print("num_new_plaintext : ", num_new_plaintext)
                print("num_new_plaintext + step: ", num_new_plaintext + step)
                sampling_from_ptx = new_masks[num_new_plaintext: num_new_plaintext + step]
                # print("sampling_from_ptx: ", sampling_from_ptx)
                print("sampling_from_ptx.shape: ", sampling_from_ptx.shape)
                num_sample_ptx = sampling_from_ptx.shape[0]
                new_latent_traces = diffusion_ema.sample(torch.from_numpy(sampling_from_ptx).to(device),
                                                         batch_size=num_sample_ptx, cond_scale=6., rescaled_phi=0.7,
                                                         clip_denoised=clip_denoised)

                collect_all_latent_traces[num_new_plaintext: num_new_plaintext + step] = new_latent_traces.cpu().numpy()

                num_new_plaintext += step
            new_latent_traces = collect_all_latent_traces
        np.save(new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(
            epochs) + "_more_shares.npy",
                new_latent_traces)
    else:
        new_latent_traces = np.load(
            new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(
                epochs) + "_more_shares.npy", )
    print("new_traces: ", new_latent_traces.shape)
    new_latent_traces = scale_transform_trace(original_X=dataloadertrain.X_profiling, new_X=new_latent_traces)

    # print(new_latent_traces)
    if decoder == False:
        new_traces = new_latent_traces
    elif decoder == True:
        save_ae_new_traces = True
        if save_ae_new_traces == True:
            ae = Autoencoder(trace_size_original, embedding_size, dims)
            ae.load_state_dict(torch.load(save_root.replace("latent_", "")+"latent_space/" + "ae_trained.pth", map_location=torch.device("cpu")))
            new_traces = ae.decode(torch.from_numpy(new_latent_traces).float()).detach()
            np.save(new_traces_root + "diffusion_new_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_more_shares.npy",new_traces)
        else:
            new_traces = np.load(new_traces_root + "diffusion_new_traces_epochs_" + dataset + "_" + leakage + "_" + str(
                epochs) + "_more_shares.npy")

if print_traces == True:
    fig_gen, ax_gen = plt.subplots(figsize=(15, 7))
    new_traces = new_traces.squeeze(1)
    x_axis = [i for i in range(new_traces.shape[1])]
    print("new_traces:", new_traces.shape)
    print("new_traces:", new_traces)
    for i, trace in enumerate(new_traces[:300, :]):
        ax_gen.plot(x_axis, trace)

    ax_gen.set_xlabel('Sample points', fontsize=20)
    ax_gen.set_ylabel('Voltage', fontsize=20)
    for label in (ax_gen.get_xticklabels() + ax_gen.get_yticklabels()):
        label.set_fontsize(15)
    plt.savefig(
        image_root + 'Generated_traces_' + dataset + "_" + leakage + "_epochs_" + str(epochs) + "_more_shares.png")

    if decoder == True:
        trace_num_sample_latent = embedding_size
        print("trace_num_sample_latent:", trace_num_sample_latent)
        print("new_latent_traces:", new_latent_traces)
        fig_new_latent, ax_new_latent = plt.subplots(figsize=(15, 7))
        x_axis = [i for i in range(trace_num_sample_latent)]
        new_latent_traces = new_latent_traces.squeeze(1)

        print("new_latent_traces:", new_latent_traces.shape)
        for i, trace in enumerate(new_latent_traces[:100, :]):
            ax_new_latent.plot(x_axis, trace)

        ax_new_latent.set_xlabel('Sample points', fontsize=20)
        ax_new_latent.set_ylabel('Voltage', fontsize=20)
        for label in (ax_gen.get_xticklabels() + ax_gen.get_yticklabels()):
            label.set_fontsize(15)
        # plt.show()
        plt.savefig(image_root + 'Latent_generated_traces_' + dataset + "_" + leakage + "_more_shares.png")
    plt.cla()

if cal_cpa == True:
    fig, ax = plt.subplots(figsize=(15, 10))
    print(new_traces.shape)
    if print_traces == False:
        new_traces = new_traces.squeeze(1)
    # print(new_traces.shape)
    # print("new_traces:", new_traces)
    # print("new_masks", new_masks.shape) #[label ^m1^m2, m1, m2], for AES_HD_ext_plaintext [pt_15,pt_11], for AES_HD_ext_sbox [sbox_inv(pt_15^k),pt_11], for AES_HD_ext_label [sbox_inv(pt_15^k)^pt_11,pt_11]
    new_plain = obtain_plaintext(dataset, new_masks, dataloadertrain.correct_key)
    # label_correct_key = aes_label_cpa_mask(dataset, new_plain, dataloadertrain.correct_key, leakage, new_masks, order = masking_order)
    #
    # print("new_plain: ", new_plain)
    new_label_mask = new_masks[:, -1]
    #
    # print("ew_label_mask == label_correct_key:", (new_label_mask == label_correct_key).all() )
    # new_traces, new_masks = unison_shuffled_copies(new_traces, new_masks)
    # if dataset == "simulated_traces_order_1":
    #     new_traces = multiply_sample_pt(new_traces, masking_order = 1)
    # elif dataset == "simulated_traces_order_2":
    #     new_traces = multiply_sample_pt(new_traces, masking_order = 2)
    used_traces = 10000
    new_traces = new_traces[:used_traces, :]
    new_label_mask = new_label_mask[:used_traces]
    # print(new_traces.shape)
    total_samplept = new_traces.shape[1]
    total_num_gen_trace = new_traces.shape[0]
    if masking_order == 0:
        label_name = "$Sbox(pt\oplus k^*)$"
        label_name_grey = r"$Sbox(pt\oplus k)$ for $k \neq k^*$"
    elif masking_order == 1:
        label_name = "$Sbox(pt\oplus k^*) \oplus m_1$"
        label_name_grey = r"$Sbox(pt\oplus k) \oplus m_1$ for $k \neq k^*$"
    elif masking_order == 2:
        label_name = "$Sbox(pt\oplus k^*) \oplus m_1 \oplus m2$"
        label_name_grey = r"$Sbox(pt\oplus k) \oplus m_1 \oplus m_2 $ for $k \neq k^*$"
    elif masking_order == 3:
        label_name = "$Sbox(pt\oplus k^*) \oplus m_1 \oplus m_2 \oplus m_3$"
        label_name_grey = r"$Sbox(pt\oplus k) \oplus m_1 \oplus m_2 \oplus m_3$ for $k \neq k^*$"
    flag = True
    for k in range(256):
        print("key: ", k)
        label_k = aes_label_cpa_mask(dataset, new_plain, k, leakage, new_masks,
                                     order=masking_order)  # new_plain is not used in AES_HD_ext
        cpa_k = cpa_method(total_samplept, total_num_gen_trace, label_k, new_traces)
        x_axis = [i for i in range(total_samplept)]
        if flag == True:
            ax.plot(x_axis, cpa_k, c="grey", label=label_name_grey)
            flag = False
        else:
            ax.plot(x_axis, cpa_k, c="grey")
    label_correct_key = aes_label_cpa_mask(dataset, new_plain, dataloadertrain.correct_key, leakage, new_masks,
                                           order=masking_order)
    cpa_k = cpa_method(total_samplept, total_num_gen_trace, label_correct_key, new_traces)
    # cpa_k = cpa_method(total_samplept, total_num_gen_trace, new_label_mask, new_traces)
    x_axis = [i for i in range(total_samplept)]
    ax.plot(x_axis, cpa_k, c="red", label=label_name)
    if dataset == "AES_HD_ext_plaintext" or dataset == "AES_HD_ext_sbox" or dataset == "AES_HD_ext_label":
        pass
    else:
        print("dataset: ", dataset)

    for shares in range(masking_order):
        cpa_k_m = cpa_method(total_samplept, total_num_gen_trace, new_masks[:, shares], new_traces)
        x_axis = [i for i in range(total_samplept)]
        if shares == 0:
            ax.plot(x_axis, cpa_k_m, c="blue", label="$m_1$")
        elif shares == 1:
            ax.plot(x_axis, cpa_k_m, c="green", label="$m_2$")
        elif shares == 2:
            ax.plot(x_axis, cpa_k_m, c="orange", label="$m_3$")
    ax.set_xlabel('Sample points', fontsize=20)
    ax.set_ylabel('(Absolute) Correlation', fontsize=20)
    for label_fig in (ax.get_xticklabels() + ax.get_yticklabels()):
        label_fig.set_fontsize(20)
    # ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 20})
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
              mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
    plt.savefig(
        image_root + 'CPA_generated_' + dataset + "_" + leakage + "_epochs_" + str(epochs) + "_more_shares_2.png")
    plt.show()
