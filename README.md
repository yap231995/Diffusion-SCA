# Diffusion_SCA



This repository developed for the paper [Creating from Noise: Trace Generations Using Diffusion Model for Side-Channel Attack](https://eprint.iacr.org/2024/167)


We uses the generative model known as Denoising Diffusion Probabilistic Model (DDPM) to generate 
In our paper, we propose two framework:
Known masked setting and unknown mask setting. <br>

## Known masked setting
### [1] Autoencoder phase
One can train the autoencoder in all the files with autoencoder in the name.<br>
The code `main_autoencoder_CW.py`, `main_autoencoder_ascadf.py`, and `main_autoencoder_ascadr.py` corresponds to training autoencoders in Chipwhisperer, ASCADf and ASCADr respectively. <br>
### [2] Diffusion training and Generative phase
`main_more_shares.py` provides the code for simulated traces as there is no autoencoder applied to it. <br>
The code `main_CW.py`, `main_ascadf.py`, and `main_ascadr.py` corresponds to training diffusion model and generating new traces with the trained diffusion model in Chipwhisperer, ASCADf and ASCADr respectively. <br>

To train, simply set 
```
training_diffusion = True
```
and to generate new traces, we set
```
sampling = True
```

## Unknown mask setting: 
The code to run the unknown mask setting is in `main_profiling.py`. 


In all the cases, simply set: 
```
want_diffusion_model = True
training_diffusion = True
```
in order to train the diffusion. <br>

One can also set `sampling = True` for using the diffusion to sample and create new dataset. 


Do cite our work at :
```
@misc{cryptoeprint:2024/167,
      author = {Trevor Yap and Dirmanto Jap},
      title = {Creating from Noise: Trace Generations Using Diffusion Model for Side-Channel Attack},
      howpublished = {Cryptology ePrint Archive, Paper 2024/167},
      year = {2024},
      note = {\url{https://eprint.iacr.org/2024/167}},
      url = {https://eprint.iacr.org/2024/167}
}
```
