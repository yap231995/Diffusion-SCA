import os
import time
from copy import deepcopy
import random
from torchvision.transforms import transforms
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataloader import ToTensor_trace, Custom_Dataset_Profiling
from src.gaussian_diffusion import GaussianDiffusion1D
from src.net import Unet1D
from src.train import train_profiling
from src.utils import NTGE_fn, find_num_traces_per_classes, create_dataset_with_min_classes, perform_attacks, \
    print_trace, scale_transform_trace
from src.template_attack import create_cov_mean_per_class, template_attack, perform_attacks_without_log
import src.template_attack as TA

dataset = "simulated_traces_order_1" #simulated_traces_order_3 #Chipwhisperer #AES_HD_ext_plaintext #AES_HD_ext_sbox #AES_HD_ext_label #latent_simulated_traces_order_0
leakage = "ID"
type_dimensionality_reduction = "None" #None #PCA
nb_traces_attacks = 1000 #simulated_data: 2000 in ASCADr i use 100000 the rest is 10000 #ASCAD_var_desync50 uses 100000
batch_size = 200 #simulated_data: 200,  CW: 50 ASCADf/r: 200 CTF2018:2000
epochs = 20 #simulated_data: 20,  CW: 100 ASCADf/r:2000 CTF2018:2000
timestamp = 4000 #for simulated: 1000, for others 4000
lr = 0.001

want_diffusion_model = True
training_diffusion = True
sampling = True
root = './'
save_root = root + 'Result/' + dataset + '_' + leakage +'_profiling/'
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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


for n_conponents in [None]: #range(8,100,8):#simulated: [None] #CW: [] #ASCAD: [24] ASCAD_variable: [16] ASCAD_desync50:[48] ASCAD_variable_desync50:[24] #range(8,50,8): #Must be divisible by 8
    embedding_size = None
    dataloadertrain = Custom_Dataset_Profiling(nb_traces_attacks, root = root, dataset = dataset, leakage = leakage,embedding_size = embedding_size, transform =  transforms.Compose([ ToTensor_trace() ]))

    # Apply AE/PCA
    if type_dimensionality_reduction == "PCA":
        dataloadertrain.PCA(n_conponents)


    dataloadertrain.split_profiling_train_val(dataloadertrain.X_profiling, dataloadertrain.Y_profiling)
    dataloadertrain.choose_phase("train_label")
    dataloaderval = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation_label")
    num_workers = 4

    trace_size = dataloadertrain.X_profiling.shape[1]
    print("trace_size: ", trace_size)
    dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers),
                   "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)}

    if leakage == "HW":
        classes = 9
    elif leakage == "ID":
        classes = 256
    model_path = save_root+dataset+"_"+leakage+"_epochs_"+str(epochs) +"_pca_"+str(n_conponents)+ "_profiling" #TODO: +"_pca_"+str(n_conponents)+ "_profiling" or + "_profiling"


    print("timestamp:", timestamp)

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        num_classes=classes
    ).to(device)
    ema_model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        num_classes=classes
    ).to(device)
    diffusion = GaussianDiffusion1D(
        model,
        device,
        seq_length = trace_size,
        timesteps = timestamp,
        # sampling_timesteps= 500,
        objective = 'pred_noise'
    )

    # print(model)
    if want_diffusion_model == True:
        if training_diffusion == True:
            from datetime import datetime
            now = datetime.now()
            dt_string = now.strftime("_%d_%m_%Y_%Hh%Mm%Ss_")
            tensorboard_root = save_root + 'tensorboard_log' + dt_string +"_pca_"+str(n_conponents)+'/'
            if not os.path.exists(tensorboard_root):
                os.mkdir(tensorboard_root)
            writer = SummaryWriter(tensorboard_root)
            start_time_training = time.time()
            ema_model,ema = train_profiling(dataloaders,diffusion,device,lr, epochs,writer, dataset, save_model_root=model_path)
            elapse_time_training= time.time() - start_time_training
            print("elapse_time for training:", elapse_time_training)
            writer.flush()
            writer.close()
        else:
            ema_model.load_state_dict(torch.load(model_path+"_ema.pth", map_location=torch.device(device)))
            model.load_state_dict(torch.load(model_path+"_original.pth", map_location=torch.device(device)))
        # print(ema_model)
        with torch.no_grad():
            diffusion_ema = GaussianDiffusion1D(
                ema_model.eval(),
                device,
                seq_length = trace_size,
                timesteps = timestamp,
                # sampling_timesteps= 500,
                objective = 'pred_noise'
            )

            new_traces = []
            new_labels = []
            #Create the labels and the shares.
            print(dataloadertrain.Y_profiling)
            num_per_class = find_num_traces_per_classes(dataloadertrain.X_profiling, dataloadertrain.Y_profiling, classes)
            min_num_class = np.min(num_per_class)
            max_num_class = np.max(num_per_class)
            min_class = np.argmin(num_per_class)
            max_class = np.argmax(num_per_class)
            print("min_num_class", min_num_class)
            print("min_class", min_class)
            print("max_num_class", max_num_class)
            print("max_class", max_class)
            print("batch size for sampling: ",int(max_num_class)*classes)
            for class_index in range(0, classes):
                # print("class_index: ", class_index)
                batch_size_sampling = int(max_num_class)
                new_plaintext_class = class_index * torch.ones((batch_size_sampling,), dtype=torch.int).to(device)
                new_labels.append(new_plaintext_class.cpu().numpy())

            new_labels = np.concatenate(new_labels, axis=0)

            #Starts sampling.
            if sampling == True:
                clip_denoised= False
                if "simulated" in dataset:
                    start_time = time.time()
                    new_latent_traces= diffusion_ema.sample(torch.from_numpy(new_labels).to(device), batch_size = batch_size_sampling*classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised= clip_denoised)
                    elapse_time_sampling = time.time() - start_time
                    print("elapse_time for sampling:", elapse_time_sampling)
                    new_latent_traces = new_latent_traces.cpu().numpy()
                else:
                    collect_all_latent_traces = np.zeros((new_labels.shape[0], 1, dataloadertrain.X_profiling.shape[1]))
                    step = 200
                    print("new_plaintext.shape: ", new_labels.shape)
                    num_new_plaintext = 0
                    start_time = time.time()
                    while num_new_plaintext <= (new_labels.shape[0]):
                        # print("num_new_plaintext: ", num_new_plaintext)
                        # print("num_new_plaintext : ", num_new_plaintext )
                        # print("num_new_plaintext + step: ", num_new_plaintext + step)
                        sampling_from_ptx = new_labels[num_new_plaintext: num_new_plaintext + step]
                        # print("sampling_from_ptx: ", sampling_from_ptx)
                        # print("sampling_from_ptx.shape: ", sampling_from_ptx.shape)
                        num_sample_ptx = sampling_from_ptx.shape[0]
                        new_latent_traces = diffusion_ema.sample(torch.from_numpy(sampling_from_ptx).to(device),
                                                                 batch_size=num_sample_ptx, cond_scale=6., rescaled_phi=0.7,
                                                                 clip_denoised=clip_denoised)

                        collect_all_latent_traces[num_new_plaintext : num_new_plaintext + step] = new_latent_traces.cpu().numpy()

                        num_new_plaintext += step
                    elapse_time_sampling = time.time() - start_time
                    print("elapse_time for sampling:", elapse_time_sampling)

                    new_latent_traces = collect_all_latent_traces
                #TODO
                np.save(new_traces_root + "elapse_time.npy", {"elapse_time_sampling": elapse_time_sampling, "elapse_time_training": elapse_time_training})
                # np.save(new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_pca_"+str(n_conponents)+".npy",new_latent_traces)
                # np.save(new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + ".npy",new_latent_traces)
            else:
                #TODO:
                new_latent_traces = np.load(new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + "_pca_"+str(n_conponents)+".npy",)
                # new_latent_traces = np.load(new_traces_root + "diffusion_latent_traces_epochs_" + dataset + "_" + leakage + "_" + str(epochs) + ".npy",)
            print("new_latent_traces: ", new_latent_traces.shape)
            # print(new_latent_traces)
            new_latent_traces = scale_transform_trace(original_X = dataloadertrain.X_profiling, new_X = new_latent_traces)

            print_traces = True
            if print_traces == True:
                print_trace(new_latent_traces, 'Generated_latent_traces_' + dataset + "_" + leakage + "_epochs_" + str(
                        epochs) + "_pca.png", image_root)
                print_trace(dataloadertrain.X_profiling, 'Original_latent_traces_' + dataset + "_" + leakage + "_epochs_" + str(
                    epochs) + "_pca.png", image_root)


    #Attack phase: TA
    padd = 0
    if dataset == "ASCAD" and type_dimensionality_reduction == None:
        padd = 2

    for type_trainning_TA in ["full_original", "reduced_original","reduce_original_plus_generated_balance"]: #TODO: "full_original","reduced_original" , "full_original_plus_generated_balance", "reduce_original_plus_generated_balance"
        if type_trainning_TA == "full_original":
            print("------" * 10)
            print(type_trainning_TA)
            if padd > 0:
                template_profiling_traces = deepcopy(dataloadertrain.X_profiling[:, padd:-padd])

            else:
                template_profiling_traces = deepcopy(dataloadertrain.X_profiling)
            template_profiling_label = deepcopy(dataloadertrain.Y_profiling)
            print("template_profiling_traces", template_profiling_traces.shape)
        elif type_trainning_TA == "reduced_original":
            print("------" * 10)
            print(type_trainning_TA)
            if padd > 0:
                template_profiling_traces, template_profiling_label, _ = create_dataset_with_min_classes(dataloadertrain.X_profiling[:,padd:-padd], dataloadertrain.Y_profiling, classes)
            else:
                template_profiling_traces, template_profiling_label, _ = create_dataset_with_min_classes(
                    dataloadertrain.X_profiling, dataloadertrain.Y_profiling, classes)
            print("template profiling traces:", template_profiling_traces.shape)
        elif type_trainning_TA == "full_original_plus_generated_balance":
            print("------" * 10)
            print(type_trainning_TA)
            if padd > 0:
                template_profiling_traces = dataloadertrain.X_profiling[:,padd:-padd]
            else:
                template_profiling_traces = dataloadertrain.X_profiling
            template_profiling_label = dataloadertrain.Y_profiling
            # print("template profiling traces:", template_profiling_traces.shape)
            # print("template profiling label:", len(template_profiling_label))

            #Find the number of traces per classes
            if padd > 0:
                num_per_class = find_num_traces_per_classes(dataloadertrain.X_profiling[:, padd:-padd], dataloadertrain.Y_profiling, classes)
            else:
                num_per_class = find_num_traces_per_classes(dataloadertrain.X_profiling,
                                                            dataloadertrain.Y_profiling, classes)
            min_num_class = np.min(num_per_class)
            max_num_class = np.max(num_per_class)
            print("min_num_class:", min_num_class)
            print("arg min_num_class:", np.argmin(num_per_class))
            print("max_num_class:", max_num_class)
            print("arg max_num_class:", np.argmax(num_per_class))
            for clas in range(classes):
                num_in_class = num_per_class[clas]
                #Generate all the new traces for this class.
                gen_clas_trace = new_latent_traces[clas*batch_size_sampling:clas*batch_size_sampling+batch_size_sampling, :].squeeze(1)
                # print("gen_clas_trace.shape: ", gen_clas_trace.shape)
                gen_clas_label = new_labels[clas*batch_size_sampling:clas*batch_size_sampling+batch_size_sampling]
                if num_in_class < max_num_class:
                    num_generated_trace_added = int(max_num_class-num_in_class)
                    #This means we need more traces to balance this class.
                    #We add the generated traces from the diffusion into the original traces for template attack.
                    if padd > 0 :
                        template_profiling_traces = np.concatenate((template_profiling_traces, gen_clas_trace[:num_generated_trace_added, padd:-padd]),axis = 0)
                    else:
                        template_profiling_traces = np.concatenate(
                            (template_profiling_traces, gen_clas_trace[:num_generated_trace_added, :]),
                            axis=0)
                    template_profiling_label = np.concatenate((template_profiling_label, gen_clas_label[:num_generated_trace_added]),axis = 0)
                    # print("batched with generated template profiling traces:", template_profiling_traces.shape)
                    # print("batched with generated template profiling label:", gen_clas_label[:num_generated_trace_added])
                    # print("batched with generated template profiling label:", gen_clas_label[:num_generated_trace_added].shape)
            print("template profiling traces:", template_profiling_traces.shape)

        elif type_trainning_TA == "reduce_original_plus_generated_balance":
            print("------" * 10)
            print(type_trainning_TA)
            print("new_latent_traces:", new_latent_traces.shape)
            X = dataloadertrain.X_profiling
            if padd > 0:
                X = dataloadertrain.X_profiling[:, padd:-padd]
            template_profiling_traces, template_profiling_label, min_num_class = create_dataset_with_min_classes(X, dataloadertrain.Y_profiling, classes)
            # print("reduced template profiling traces:", template_profiling_traces.shape)
            # print("reduced template profiling label:", template_profiling_label.shape)
            # print("min_num_class:", min_num_class)
            for clas in range(classes):
                # min_num_class = 12294
                num_in_class = num_per_class[clas]
                # Generate all the new traces for this class.
                if padd > 0:
                    gen_clas_trace = new_latent_traces[
                                     clas * batch_size_sampling:clas * batch_size_sampling + batch_size_sampling, padd:-padd].squeeze(1)
                else:
                    gen_clas_trace = new_latent_traces[clas * batch_size_sampling:clas * batch_size_sampling + batch_size_sampling, :].squeeze(1)
                gen_clas_label = new_labels[clas * batch_size_sampling:clas * batch_size_sampling + batch_size_sampling]
                template_profiling_traces = np.concatenate((template_profiling_traces, gen_clas_trace[:min_num_class, :]), axis=0)
                template_profiling_label = np.concatenate(
                    (template_profiling_label, gen_clas_label[:min_num_class]), axis=0)
                # print("label added:", gen_clas_label[:min_num_class])
                # print("batched with generated template profiling traces:", template_profiling_traces.shape)
                # print("batched with generated template profiling label:", template_profiling_label.shape)
            print("template profiling traces:", template_profiling_traces.shape)
        #Build Template
        build_template = True
        if padd >0:
            template_attack_traces = deepcopy(dataloadertrain.X_attack[:, padd:-padd])
        else:
            template_attack_traces = deepcopy(dataloadertrain.X_attack)
        num_features = template_attack_traces.shape[1]
        print("num_features", num_features)
        if build_template == True:
            determinant_classes, mean_classes, cov_classes = create_cov_mean_per_class(
                template_profiling_traces, template_profiling_label, num_features,
                classes)
            np.save(save_root+"ta_stuff_"+type_trainning_TA+"_pca_"+str(n_conponents)+".npy", {"mean_classes":mean_classes, "cov_classes":cov_classes, "determinant_classes":determinant_classes})
        else:
            ta_stuff = np.load(save_root+"ta_stuff_"+type_trainning_TA+"_pca_"+str(n_conponents)+".npy", allow_pickle = True).item()
            mean_classes = ta_stuff["mean_classes"]
            cov_classes = ta_stuff["cov_classes"]
            determinant_classes = ta_stuff["determinant_classes"]
        print("template_attack_traces.shape: ",template_attack_traces.shape)
        predictions_ta_log = template_attack(template_attack_traces[:2*nb_traces_attacks, :], classes, determinant_classes, #TODO
                                                  mean_classes, cov_classes)

        #Attack with Template
        GE, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log, dataloadertrain.plt_attack,
                                                 dataloadertrain.correct_key,
                                                 leakage,
                                                 nb_attacks=50,
                                                 shuffle=True, dataset=dataset)

        NTGE= NTGE_fn(GE)
        print("type_trainning_TA:", type_trainning_TA)
        print("GE:", GE)
        print("NTGE:", NTGE)
        np.save(save_root+"misc_"+type_trainning_TA+"_pca_"+str(n_conponents)+".npy",{"GE":GE,"NTGE": NTGE})







