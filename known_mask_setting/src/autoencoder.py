from torch import nn
import torch.nn.functional as F
import torch
import random

def get_hyperparameters_mlp(regularization=False, max_dense_layers=8):
    if regularization:
        return {
            "batch_size": random.randrange(100, 1100, 100),
            "layers": random.randrange(1, max_dense_layers + 1, 1),
            "neurons": random.choice([10, 20, 50, 100, 200, 300, 400, 500]),
            "activation": random.choice(["relu", "selu"]),
            "learning_rate": random.choice([0.005, 0.001, 0.0005, 0.0001]),
            "optimizer": random.choice(["Adam", "RMSprop"]),
            "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
            "regularization": random.choice(["l1", "l2", "dropout"]),
            "l1": random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
            "l2": random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
            "dropout": random.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        }
    else:
        return {
            "batch_size": random.randrange(100, 1100, 100),
            "layers": random.choice([1, 2, 3, 4]),
            "neurons": random.choice([10, 20, 50, 100, 200, 300, 400, 500]),
            "activation": random.choice(["relu", "selu"]),
            "learning_rate": random.choice([0.005, 0.001, 0.0005, 0.0001]),
            "optimizer": random.choice(["Adam", "RMSprop"]),
            "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
            "regularization": random.choice(["none"])
        }




def get_hyperparemeters_cnn(regularization=False):
    hyperparameters = {}
    hyperparameters_mlp = get_hyperparameters_mlp(regularization=regularization, max_dense_layers=4)
    for key, value in hyperparameters_mlp.items():
        hyperparameters[key] = value

    conv_layers = random.choice([1, 2, 3, 4])
    kernels = []
    strides = []
    filters = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []
    pooling_type = random.choice(["Average", "Max"])

    for conv_layer in range(1, conv_layers + 1):
        kernel = random.randrange(26, 52, 2)
        kernels.append(kernel)
        strides.append(int(kernel / 2))
        if conv_layer == 1:
            filters.append(random.choice([4, 8, 12, 16]))
        else:
            filters.append(filters[conv_layer - 2] * 2)
        pool_size = random.choice([2, 4, 6, 8, 10])
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)

    hyperparameters["conv_layers"] = conv_layers
    hyperparameters["kernels"] = kernels
    hyperparameters["strides"] = strides
    hyperparameters["filters"] = filters
    hyperparameters["pooling_sizes"] = pooling_sizes
    hyperparameters["pooling_strides"] = pooling_strides
    hyperparameters["pooling_types"] = pooling_types

    return hyperparameters



class Autoencoder(nn.Module):

    def __init__(self, inputsize, embedding_size, dims):
        super().__init__()
        self.embedding_size = embedding_size

        self.inputsize = inputsize

        # self.enc1 = nn.Linear(inputsize, 50)
        # self.enc2 = nn.Linear(50, self.embedding_size)
        # self.dec1 = nn.Linear(self.embedding_size, 50)
        # self.dec2 = nn.Linear(50, inputsize)

        dims.append(self.embedding_size)
        encmodules = []
        encmodules.append(nn.Linear(inputsize, dims[0]))
        for index in range(len(dims ) -1):
            encmodules.append(nn.ReLU(True))
            encmodules.append(nn.Linear(dims[index], dims[index +1]))

        self.encode = nn.Sequential(*encmodules)

        decmodules = []
        for index in range(len(dims) - 1, 0, -1):
            decmodules.append(nn.Linear(dims[index], dims[index -1]))
            decmodules.append(nn.ReLU(True))
        decmodules.append(nn.Linear(dims[0], inputsize))
        self.decoder = nn.Sequential(*decmodules)

    # def encode(self, x):
    #     #print("ENCODER!!")
    #     x = self.enc1(x)
    #     x = F.relu(x)
    #     x = self.enc2(x)
    #     x = F.relu(x)
    #     return x
    #
    def decode(self, x):
        #print("DECODER!!")
        return (self.decoder(x))
        #return F.sigmoid(self.decoder(x))

    def forward(self,x):
        x =self.encode(x)
        x = self.decode(x)
        return x




# class Autoencoder(nn.Module):
#
#     def __init__(self, inputsize, embedding_size):
#         super().__init__()
#         self.embedding_size = embedding_size
#
#         self.inputsize = inputsize
#
#         self.conv1_1 = nn.Conv1d(1, 256, 2, padding='same')
#         self.bn1_1 = nn.BatchNorm1d(256)
#         self.conv1_2 = nn.Conv1d(256, 256, 2, padding='same')
#         self.bn1_2 = nn.BatchNorm1d(256)
#         self.conv1_3 = nn.Conv1d(256, 256, 2, padding='same')
#         self.bn1_3 = nn.BatchNorm1d(256)
#         self.conv1_4 = nn.Conv1d(256, 256, 2, padding='same')
#         self.bn1_4 = nn.BatchNorm1d(256)
#         self.conv1_5 = nn.Conv1d(256, 256, 2, padding='same')
#         self.bn1_5 = nn.BatchNorm1d(256)
#         self.conv1_6 = nn.Conv1d(256, 256, 2, padding='same')
#         self.bn1_6 = nn.BatchNorm1d(256)
#
#         self.pool_1 = nn.MaxPool1d(5,5)
#
#         self.conv2_1 = nn.Conv1d(256, 128, 2, padding='same')
#         self.bn2_1 = nn.BatchNorm1d(128)
#         self.conv2_2 = nn.Conv1d(128, 128, 2, padding='same')
#         self.bn2_2 = nn.BatchNorm1d(128)
#         self.conv2_3 = nn.Conv1d(128, 128, 2, padding='same')
#         self.bn2_3 = nn.BatchNorm1d(128)
#         self.conv2_4 = nn.Conv1d(128, 128, 2, padding='same')
#         self.bn2_4 = nn.BatchNorm1d(128)
#
#         self.pool_2 = nn.MaxPool1d(2, 2)
#
#         self.conv3_1 = nn.Conv1d(128, 64, 2, padding='same')
#         self.bn3_1 = nn.BatchNorm1d(64)
#         self.conv3_2 = nn.Conv1d(64, 64, 2, padding='same')
#         self.bn3_2 = nn.BatchNorm1d(64)
#         self.conv3_3 = nn.Conv1d(64, 64, 2, padding='same')
#         self.bn3_3 = nn.BatchNorm1d(64)
#         self.conv3_4 = nn.Conv1d(64, 64, 2, padding='same')
#         self.bn3_4 = nn.BatchNorm1d(64)
#         self.pool_3 = nn.MaxPool1d(2, 2)
#
#         flatten_size = ((self.inputsize//2)//2)//5*64
#         #print("flatten_size: ", flatten_size)
#         self.fc1 = nn.Linear(flatten_size, self.embedding_size)
#         self.fc2 = nn.Linear(self.embedding_size, flatten_size)
#         ## decoder layers ##
#         self.upsample_1 = nn.Upsample(scale_factor=2)
#         self.deconv1_1 = nn.ConvTranspose1d(64, 64, 1)
#         self.dbn1_1 = nn.BatchNorm1d(64)
#         self.deconv1_2 = nn.ConvTranspose1d(64, 64, 1)
#         self.dbn1_2 = nn.BatchNorm1d(64)
#         self.deconv1_3 = nn.ConvTranspose1d(64, 64, 1)
#         self.dbn1_3 = nn.BatchNorm1d(64)
#         self.deconv1_4 = nn.ConvTranspose1d(64, 64, 1)
#         self.dbn1_4 = nn.BatchNorm1d(64)
#
#         self.upsample_2 = nn.Upsample(scale_factor=2)
#
#         self.deconv2_1 = nn.ConvTranspose1d(64, 128, 1)
#         self.dbn2_1 = nn.BatchNorm1d(128)
#         self.deconv2_2 = nn.ConvTranspose1d(128, 128, 1)
#         self.dbn2_2 = nn.BatchNorm1d(128)
#         self.deconv2_3 = nn.ConvTranspose1d(128, 128, 1)
#         self.dbn2_3 = nn.BatchNorm1d(128)
#         self.deconv2_4 = nn.ConvTranspose1d(128, 128, 1)
#         self.dbn2_4 = nn.BatchNorm1d(128)
#
#         self.upsample_3 = nn.Upsample(scale_factor=5)
#
#         self.deconv3_1 = nn.ConvTranspose1d(128, 256, 1)
#         self.dbn3_1 = nn.BatchNorm1d(256)
#         self.deconv3_2 = nn.ConvTranspose1d(256, 256, 1)
#         self.dbn3_2 = nn.BatchNorm1d(256)
#         self.deconv3_3 = nn.ConvTranspose1d(256, 256, 1)
#         self.dbn3_3 = nn.BatchNorm1d(256)
#         self.deconv3_4 = nn.ConvTranspose1d(256, 256, 1)
#         self.dbn3_4 = nn.BatchNorm1d(256)
#         self.deconv3_5 = nn.ConvTranspose1d(256, 256, 1)
#         self.dbn3_5 = nn.BatchNorm1d(256)
#         self.deconv3_6 = nn.ConvTranspose1d(256, 1, 1)
#         # self.dbn3_6 = nn.BatchNorm1d(256)
#
#     def encode(self, x):
#         #print("ENCODER!!")
#
#         #print(x.shape)
#         x = self.bn1_1(F.selu(self.conv1_1(x)))
#         x = self.bn1_2(F.selu(self.conv1_2(x)))
#
#         x = self.bn1_3(F.selu(self.conv1_3(x)))
#         x = self.bn1_4(F.selu(self.conv1_4(x)))
#         x = self.bn1_5(F.selu(self.conv1_5(x)))
#         x = self.bn1_6(F.selu(self.conv1_6(x)))
#
#         #print(x.shape)
#         x = self.pool_1(x)
#
#         #print(x.shape)
#         x = self.bn2_1(F.selu(self.conv2_1(x)))
#         x = self.bn2_2(F.selu(self.conv2_2(x)))
#         x = self.bn2_3(F.selu(self.conv2_3(x)))
#         x = self.bn2_4(F.selu(self.conv2_4(x)))
#
#         #print(x.shape)
#         x = self.pool_2(x)
#
#         #print(x.shape)
#         x = self.bn3_1(F.selu(self.conv3_1(x)))
#         x = self.bn3_2(F.selu(self.conv3_2(x)))
#         x = self.bn3_3(F.selu(self.conv3_3(x)))
#         x = self.bn3_4(F.selu(self.conv3_4(x)))
#         x = self.pool_3(x)
#         #print(x.shape)
#         x = x.view(x.shape[0], -1)
#
#         #print(x.shape)
#         x = F.selu(self.fc1(x))
#         #print(x.shape)
#         return x
#
#     def decode(self, x):
#
#         #print("DECODER!!")
#         #print(x.shape)
#         x = F.selu(self.fc2(x))
#         #print(x.shape)
#         x = x.view(x.shape[0], 64, -1)
#         #print(x.shape)
#         x = self.upsample_1(x)
#         #print(x.shape)
#         x = self.dbn1_1(F.selu(self.deconv1_1(x)))
#         #print(x.shape)
#         x = self.dbn1_2(F.selu(self.deconv1_2(x)))
#         #print(x.shape)
#         x = self.dbn1_3(F.selu(self.deconv1_3(x)))
#         #print(x.shape)
#         x = self.dbn1_4(F.selu(self.deconv1_4(x)))
#         #print(x.shape)
#         x = self.upsample_2(x)
#         #print("upsample_2: ",x.shape)
#         x = self.dbn2_1(F.selu(self.deconv2_1(x)))
#         x = self.dbn2_2(F.selu(self.deconv2_2(x)))
#         x = self.dbn2_3(F.selu(self.deconv2_3(x)))
#         x = self.dbn2_4(F.selu(self.deconv2_4(x)))
#         #print(x.shape)
#
#         x = self.upsample_3(x)
#         #print("upsample_3: ",x.shape)
#         x = self.dbn3_1(F.selu(self.deconv3_1(x)))
#         x = self.dbn3_2(F.selu(self.deconv3_2(x)))
#         x = self.dbn3_3(F.selu(self.deconv3_3(x)))
#         x = self.dbn3_4(F.selu(self.deconv3_4(x)))
#         x = self.dbn3_5(F.selu(self.deconv3_5(x)))
#         x = self.deconv3_6(x)
#         x = F.sigmoid(x)
#         #print(x.shape)
#         return x
#
#     def forward(self,x):
#         x =self.encode(x)
#         x = self.decode(x)
#         return x





class VAE(nn.Module):
    def __init__(self, inputsize, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.inputsize = inputsize

        h_channel = inputsize
        dims = []
        for i in range(4):
            h_channel = h_channel//2
            dims.append(h_channel)
        self.enc1 = nn.Linear(inputsize, dims[0])
        self.enc2 = nn.Linear(dims[0], dims[1])
        self.enc3 = nn.Linear(dims[1], dims[2])
        self.enc4 = nn.Linear(dims[2], dims[3])
        self.mu_linear = nn.Linear(dims[3], embedding_size)
        self.sigma_linear = nn.Linear(dims[3], embedding_size)


        self.dec1 = nn.Linear(embedding_size, dims[3])
        self.dec2 = nn.Linear(dims[3], dims[2])
        self.dec3 = nn.Linear(dims[2], dims[1])
        self.dec4 = nn.Linear(dims[1], dims[0])
        self.dec5 = nn.Linear(dims[0], inputsize)


    def encode(self, x):
        h = F.selu(self.enc1(x))
        h = F.selu(self.enc2(h))
        h = F.selu(self.enc3(h))
        h = F.selu(self.enc4(h))
        mu = self.mu_linear(h)
        sigma = self.sigma_linear(h)
        return mu,sigma
    def decode(self,x):
        h = F.selu(self.dec1(x))
        h = F.selu(self.dec2(h))
        h = F.selu(self.dec3(h))
        h = F.selu(self.dec4(h))
        h = torch.sigmoid(self.dec5(h))
        return h

    def forward(self, x):
        mu,sigma= self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma

