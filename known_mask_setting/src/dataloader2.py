import os

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils2 import load_chipwhisperer, generate_traces, calculate_HW, load_aes_hd_ext, load_latent_dataset, \
    AES_Sbox_inv, load_ascad, load_ctf, load_ecc
import torch
from sklearn.decomposition import PCA

class Custom_Dataset(Dataset):
    def __init__(self, root = './', dataset = "Chipwhisperer", leakage = "HW",transform = None, embedding_size = None):
        if dataset == 'simulated_traces_order_0' or dataset == 'simulated_traces_order_1' or dataset == 'simulated_traces_order_2' or dataset == 'simulated_traces_order_3':
            data_root = './Dataset/' + dataset + '/'
            if not os.path.exists(data_root):
                os.mkdir(data_root)
            if dataset == 'simulated_traces_order_0':
                self.order = 0
            elif dataset == 'simulated_traces_order_1':
                self.order = 1
            elif dataset == 'simulated_traces_order_2':
                self.order = 2
            elif dataset == 'simulated_traces_order_3':
                self.order = 3
            save_data = False
            self.correct_key = 0x03
            trace_size = 24 #24,64,88,128,216,256, 512
            if save_data == True:
                self.X_profiling, self.Y_profiling, self.plt_profiling, self.masking_profiling = generate_traces(n_traces=14000, n_features=trace_size, order=self.order)
                self.X_attack, self.Y_attack, self.plt_attack, self.masking_attack = generate_traces(n_traces=30000, n_features=trace_size, order=self.order)
                np.save(data_root + "X_profiling.npy", self.X_profiling)
                np.save(data_root + "Y_profiling.npy", self.Y_profiling)
                np.save(data_root + "plt_profiling.npy", self.plt_profiling)
                np.save(data_root + "X_attack.npy", self.X_attack)
                np.save(data_root + "Y_attack.npy", self.Y_attack)
                np.save(data_root + "plt_attack.npy", self.plt_attack)
                np.save(data_root + "masking_profiling.npy", self.masking_profiling)
                np.save(data_root + "masking_attack.npy", self.masking_attack)
                if leakage == 'HW':
                    self.Y_profiling = np.array(calculate_HW(self.Y_profiling))
                    self.Y_attack = np.array(calculate_HW(self.Y_attack))
            else:
                self.X_profiling = np.load(data_root + "X_profiling.npy")
                self.Y_profiling = np.load(data_root + "Y_profiling.npy")
                self.plt_profiling = np.load(data_root + "plt_profiling.npy")
                self.X_attack = np.load(data_root + "X_attack.npy")
                self.Y_attack = np.load(data_root + "Y_attack.npy")
                self.plt_attack = np.load(data_root + "plt_attack.npy")
                if os.path.exists(data_root+"masking_attack.npy") and os.path.exists(data_root+"masking_profiling.npy") :
                    self.masking_profiling = np.load(data_root + "masking_profiling.npy")
                    self.masking_attack = np.load(data_root + "masking_attack.npy")
                if leakage == 'HW':
                    self.Y_profiling = np.array(calculate_HW(self.Y_profiling))
                    self.Y_attack = np.array(calculate_HW(self.Y_attack))
        elif dataset == 'ECC':
            data_root = 'Dataset/ECC/'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_ecc(root + data_root + '/',
                                                                                        leakage_model=leakage)
            self.masking_profiling = [self.Y_profiling, ]
            self.masking_attack = [self.Y_attack, ]
        elif dataset == 'Chipwhisperer':
            data_root = 'Dataset/Chipwhisperer/'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack), self.correct_key = load_chipwhisperer(root + data_root + '/', leakage_model=leakage)
            self.masking_profiling = [self.Y_profiling,]
            self.masking_attack = [self.Y_attack,]
            self.order = 0
        elif dataset == 'AES_HD_ext' or dataset == 'AES_HD_ext_plaintext' or dataset == "AES_HD_ext_sbox":
            data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_aes_hd_ext(root + data_root, leakage_model=leakage,
                                                                      train_begin=0, train_end=45000,
                                                                      test_begin=0,
                                                                      test_end=10000)
            self.X_profiling = np.pad(self.X_profiling, ((0,0),(7,7)),  'constant', constant_values = (0,0))
            print("self.X_profiling.shape after pad: ",self.X_profiling.shape)
            self.X_attack = np.pad(self.X_attack, ((0,0),(7,7)),  'constant', constant_values = (0,0))
            print("self.X_attack.shape after pad: ", self.X_attack.shape)
            self.masking_profiling = [self.Y_profiling, ]
            self.masking_attack = [self.Y_attack, ]
            self.order = 1
        elif dataset == "latent_simulated_traces_order_0":
            if embedding_size == None:
                raise("NEED embbeding_size for dataset using latent traces")

            data_root = root + 'Result/' + dataset.replace("latent_","") + '_' + leakage +'/'+ 'latent_space/' + 'latent_dataset_emb_' + str(embedding_size) + '/'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), \
            (self.plt_profiling, self.plt_attack), self.correct_key = load_latent_dataset(data_root, leakage_model=leakage)
        print("The dataset we using: ", data_root)
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.scaler = MinMaxScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)
        self.dataset = dataset

        # self.scaler_std_latent = StandardScaler()
        # self.scaler_latent = MinMaxScaler()
        # self.X_profiling_train, self.X_profiling_val, self.Y_profiling_train, self.Y_profiling_val = train_test_split(self.X_profiling,self.Y_profiling,test_size=0.1,random_state=0)

        # print("X_profiling:", self.X_profiling)
        print("X_profiling max:", np.max(self.X_profiling))
        print("X_profiling min:", np.min(self.X_profiling))
        # print("plt_profiling:", self.plt_profiling)
    def apply_MinMaxScaler(self):
        self.X_profiling = self.scaler.fit_transform(self.X_profiling)
        self.X_attack = self.scaler.transform(self.X_attack)
        print("After minmaxscaler X_profiling max:", np.max(self.X_profiling))
        print("After minmaxscaler X_profiling min:", np.min(self.X_profiling))


    def choose_phase(self,phase):
        if phase == 'train_label':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling, 1), self.Y_profiling, self.Y_profiling
            # elif phase == 'validation':
            #     self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling_val, 1), self.Y_profiling_val
        elif phase == 'test_label':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_attack, 1), self.Y_attack, self.Y_attack

        if phase == 'train_plaintext':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling, 1), self.Y_profiling, self.plt_profiling
            # if self.dataset == "AES_HD_ext_plaintext":
            #     self.Plaintext = [self.plt_profiling[:,15], self.plt_profiling[:,11]]
        # elif phase == 'validation':
        #     self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling_val, 1), self.Y_profiling_val
        elif phase == 'test_plaintext':
            self.X, self.Y, self.Plaintext =np.expand_dims(self.X_attack, 1), self.Y_attack, self.plt_attack
        elif phase == 'train_more_shares':

            self.X, self.Y= np.expand_dims(self.X_profiling, 1), self.Y_profiling,
            print("self.X: ", self.X.shape)
            print("self.Y: ", self.Y.shape)
            if self.dataset == "AES_HD_ext_plaintext":
                self.Plaintext = np.concatenate([np.expand_dims(self.plt_profiling[:,15], 1), np.expand_dims(self.plt_profiling[:,11],1)], axis = 1)
            elif self.dataset == "AES_HD_ext_sbox":
                sbox_inv_label = np.zeros(self.plt_profiling.shape[0])
                print("sbox_inv_label.shape: ", sbox_inv_label.shape)
                for i in range(self.plt_profiling.shape[0]):
                    sbox_inv_label[i] = AES_Sbox_inv[self.plt_profiling[i, 15] ^ self.correct_key]
                self.Plaintext = np.concatenate([np.expand_dims(sbox_inv_label, 1), np.expand_dims(self.plt_profiling[:,11],1)], axis = 1)
            elif self.dataset == "AES_HD_ext_label":
                self.Plaintext = np.concatenate([np.expand_dims(self.Y_profiling, 1), np.expand_dims(self.plt_profiling[:,11],1)], axis = 1)
            else:
                all_mask = np.zeros((self.Y.shape[0], self.order+1))
                for share in range(self.order + 1):
                    all_mask[:, share] = self.masking_profiling[share]
                # all_mask[:, self.order] = self.plt_profiling #This doesn't work for order 2 and order 3.
                self.Plaintext = all_mask #In training, we use self.Plaintext as the conditional labels.

        elif phase == 'train_latent':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.latent_traces_profiling, 1), self.Y_profiling, self.plt_profiling
        elif phase == 'test_latent':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.latent_traces_attack, 1), self.Y_attack, self.plt_attack

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        plaintext = self.Plaintext[idx]
        sample = {'trace': trace, 'sensitive': sensitive, 'plaintext': plaintext}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample




class Custom_Dataset_Profiling(Dataset):
    def __init__(self,nb_traces_attacks, root = './', dataset = "Chipwhisperer", leakage = "HW",transform = None, embedding_size = None):
        self.nb_traces_attacks = nb_traces_attacks
        if dataset == 'Chipwhisperer':
            data_root = 'Dataset/Chipwhisperer/'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack), self.correct_key = load_chipwhisperer(root + data_root + '/', leakage_model=leakage)
            self.masking_profiling = [self.Y_profiling,]
            self.masking_attack = [self.Y_attack,]
            self.order = 0
        elif dataset == 'AES_HD_ext' or dataset == 'AES_HD_ext_plaintext' or dataset == "AES_HD_ext_sbox":
            data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_aes_hd_ext(root + data_root, leakage_model=leakage,
                                                                      train_begin=0, train_end=45000,
                                                                      test_begin=0,
                                                                      test_end=10000)
            #TODO: I remove the padding for profiling
            # self.X_profiling = np.pad(self.X_profiling, ((0,0),(7,7)),  'constant', constant_values = (0,0))
            # print("self.X_profiling.shape after pad: ",self.X_profiling.shape)
            # self.X_attack = np.pad(self.X_attack, ((0,0),(7,7)),  'constant', constant_values = (0,0))
            # print("self.X_attack.shape after pad: ", self.X_attack.shape)
            self.masking_profiling = [self.Y_profiling, ]
            self.masking_attack = [self.Y_attack, ]
            self.order = 1
        elif dataset == 'ECC':
            data_root = 'Dataset/ECC/databaseEdDSA.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_ecc(root + data_root,
                                                                                        leakage_model=leakage)
            self.order = 0
        elif dataset == "ASCAD":
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=self.nb_traces_attacks)
            self.X_profiling_orig = self.X_profiling
            self.X_attack_orig = self.X_attack
            self.X_profiling = np.pad(self.X_profiling, ((0, 0), (2, 2)), 'constant', constant_values=(0, 0))
            self.X_attack = np.pad(self.X_attack, ((0, 0), (2, 2)), 'constant', constant_values=(0, 0))
            self.order = 1
        elif dataset == "ASCAD_variable":
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_variable.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=self.nb_traces_attacks)
        elif dataset == "ASCAD_desync50":
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_desync50.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=self.nb_traces_attacks)
        elif dataset == "ASCAD_desync100":
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_desync100.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=self.nb_attack_trace)
        elif dataset == "ASCAD_variable_desync50":
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_variable_desync50.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=100000, test_begin=0,
                test_end=self.nb_traces_attacks)
        elif dataset == "ASCAD_variable_desync100":
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_variable_desync100.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=self.nb_traces_attacks)
        elif dataset == 'CTF2018':
            byte = 0
            data_root = 'Dataset/CTF2018/ches_ctf.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack), self.correct_key = load_ctf(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=self.nb_traces_attacks)
        print("The dataset we using: ", data_root)
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.scaler = MinMaxScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)
        self.dataset = dataset

        # self.scaler_std_latent = StandardScaler()
        # self.scaler_latent = MinMaxScaler()
        # print("X_profiling:", self.X_profiling)
        print("X_profiling max:", np.max(self.X_profiling))
        print("X_profiling min:", np.min(self.X_profiling))
        # print("plt_profiling:", self.plt_profiling)
    def apply_MinMaxScaler(self):
        self.X_profiling = self.scaler.fit_transform(self.X_profiling)
        self.X_attack = self.scaler.transform(self.X_attack)
        print("After minmaxscaler X_profiling max:", np.max(self.X_profiling))
        print("After minmaxscaler X_profiling min:", np.min(self.X_profiling))

    def PCA(self, n_components):
        self.PCA = PCA(n_components=n_components)
        self.PCA.fit(self.X_profiling)
        self.X_profiling =self.PCA.transform(self.X_profiling)
        self.X_attack = self.PCA.transform(self.X_attack)
        print("PCA X_profiling ", self.X_profiling.shape)
        print("PCA X_attack ", self.X_attack.shape)

    def split_profiling_train_val(self, X, Y):
        self.X_profiling_train, self.X_profiling_val, self.Y_profiling_train, self.Y_profiling_val = train_test_split(
            X, Y, test_size=0.1, random_state=0)

    def choose_phase(self,phase):
        if phase == 'train_label':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling_train, 1), self.Y_profiling_train, self.Y_profiling_train
        elif phase == 'validation_label':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling_val, 1), self.Y_profiling_val,self.Y_profiling_val
        elif phase == 'test_label':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_attack, 1), self.Y_attack, self.Y_attack

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        plaintext = self.Plaintext[idx]
        sample = {'trace': trace, 'sensitive': sensitive, 'plaintext': plaintext}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample




class ToTensor_trace(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, label, plaintext= sample['trace'], sample['sensitive'], sample['plaintext']

        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).long(), torch.from_numpy(np.array(plaintext)).long()


class Classifier_Dataset(Dataset):
    def __init__(self,X_profiling,Y_profiling,X_attack,Y_attack,plt_attack,correct_key, transform=None):
        self.X_profiling = X_profiling
        self.Y_profiling = Y_profiling
        self.X_attack = X_attack
        self.Y_attack = Y_attack
        self.transform = transform
        self.split_profiling_train_val(self.X_profiling, self.Y_profiling)
        self.correct_key = correct_key
        self.plt_attack = plt_attack

    def split_profiling_train_val(self, X, Y):
        self.X_profiling_train, self.X_profiling_val, self.Y_profiling_train, self.Y_profiling_val = train_test_split(
            X, Y, test_size=0.1, random_state=0)

    def choose_phase(self,phase):
        if phase == 'train':
            self.X, self.Y = np.expand_dims(self.X_profiling_train, 1), self.Y_profiling_train
        elif phase == 'validation':
            self.X, self.Y = np.expand_dims(self.X_profiling_val, 1), self.Y_profiling_val
        elif phase == 'test':
            self.X, self.Y = np.expand_dims(self.X_attack, 1), self.Y_attack

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        # plaintext = self.Plaintext[idx]
        sample = {'trace': trace, 'sensitive': sensitive}  # , 'plaintext': plaintext}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor_trace_classifier(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, label = sample['trace'], sample['sensitive']#, sample['plaintext']

        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).long()#, torch.from_numpy(np.array(plaintext)).long()