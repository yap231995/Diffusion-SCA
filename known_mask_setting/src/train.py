


import os
import copy
import numpy as np
import torch
from torch import optim, nn
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.net import EMA

def train(dataloaders,diffusion, device,lr, epochs,dataset, save_model_root):
    dataloader = dataloaders["train"]

    optimizer = optim.AdamW(diffusion.model.parameters(), lr = lr)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(diffusion.model).eval().requires_grad_(False)

    for epoch in range(epochs):
        #print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0
        total_iteration = 0
        for i, (traces, labels,plaintext) in enumerate(pbar):
            traces = traces.to(device)
            labels = labels.to(device)
            plaintext = plaintext.to(device)
            #print("traces: ", traces.shape)
            # if dataset == "AES_HD_ext":
            #     loss = diffusion(traces, labels)
            # else:
            #print(traces.shape)
            #print(plaintext.shape)
            loss = diffusion(traces, plaintext)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, diffusion.model)
            total_iteration+=1
            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        #writer.add_scalar("Loss/train", epoch_loss/total_iteration, epoch)
        print("MSE: {} Epoch:{}/{}".format(epoch_loss/total_iteration, epoch, epochs))
        # if epoch % 50 == 0:
        #     torch.save(diffusion.model.state_dict(), save_model_root+"_epochs_"+str(epoch)+"_original.pth")
        #     torch.save(ema_model.state_dict(), save_model_root+"_epochs_"+str(epoch)+"_ema.pth")
    torch.save(diffusion.model.state_dict(), save_model_root + "_original.pth")
    torch.save(ema_model.state_dict(), save_model_root + "_ema.pth")
    return ema_model, ema


def train_ae(dataloaders,ae, epochs_ae, device, lr_ae,  latent_space_root, kl_weights = 1):
    dataloader = dataloaders["train"]
    #writer_ae = SummaryWriter(latent_space_root + "ae_training/")
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr_ae)
    criterion = nn.MSELoss()#nn.BCELoss(reduction = "sum") #nn.MSELoss()
    # epochs_ae = 1
    for epoch in range(epochs_ae):
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, disable=True)
        epoch_loss = 0
        total_iteration = 0
        for i, (traces, labels,plaintext) in enumerate(pbar):
            traces = traces.to(device)
            # labels = labels.to(device)
            # plaintext = plaintext.to(device)
            # print("original traces: ", traces.shape)
            reconstructed_traces = ae(traces)
            # print(reconstructed_traces.shape)
            # print(traces.shape)
            loss = criterion(traces,reconstructed_traces)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iteration+=1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        #writer_ae.add_scalar("Loss/train", epoch_loss/total_iteration, epoch)
        print("loss: {} Epoch:{}/{}".format(epoch_loss/total_iteration, epoch, epochs_ae))
    #writer_ae.flush()
    #writer_ae.close()
    torch.save(ae.state_dict(), latent_space_root + "ae_trained.pth")

def train_vae(dataloaders, vae, epochs_ae, device, lr_ae, latent_space_root, kl_weights=1, ):
    dataloader = dataloaders["train"]
    #writer_ae = SummaryWriter(latent_space_root + "ae_training/")
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr_ae)
    criterion = nn.CrossEntropyLoss(reduction = "sum")  # nn.BCELoss(reduction = "sum") #nn.MSELoss() #.nn.CrossEntropyLoss
    for epoch in range(epochs_ae):
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0
        total_iteration = 0
        for i, (traces, labels, plaintext) in enumerate(pbar):
            traces_dim = traces.shape[2]
            # print(traces.shape)
            traces = traces.to(device)
            reconstructed_traces,mu,sigma = vae(traces)
            # print(reconstructed_traces.shape)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            reconstruction_loss = criterion(traces, reconstructed_traces)
            loss = reconstruction_loss + (
                        kl_weights / traces_dim) * kl_div  # note that BCE scales with the dimension that is why we need to scale it accordingly

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iteration += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        #writer_ae.add_scalar("Loss/train", epoch_loss / total_iteration, epoch)
        print("loss: {} Epoch:{}/{}".format(epoch_loss / total_iteration, epoch, epochs_ae))
    #writer_ae.flush()
    #writer_ae.close()
    torch.save(vae.state_dict(), latent_space_root + "vae_trained.pth")

        #trace_reconstructed, mu, sigma = ae(traces)

            # print("trace_reconstructed: ", trace_reconstructed.shape)
            # reconstruction_loss = criterion(trace_reconstructed, traces)
            # kl_div = -0.5*torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2) - sigma.pow(2))
            # print("kl_div", kl_div)
            # print("reconstruction_loss", reconstruction_loss)
            # loss = reconstruction_loss + (kl_weights/image_dim)*kl_div #note that BCE scales with the dimension that is why we need to scale it accordingly
