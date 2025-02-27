import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import h5py
import random
import math
from tqdm import tqdm
AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
AES_Sbox_inv =  np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

def HW(s):
    return bin(s).count("1")

def multiply_sample_pt(traces_consider, masking_order = 0):
    if masking_order == 0:
        return traces_consider
    elif masking_order == 1:
        extended_traces = np.zeros((traces_consider.shape[0], traces_consider.shape[1] * (traces_consider.shape[1] - 1) // 2))
        counter = 0

        for i in range(traces_consider.shape[1]):
            for j in range(i + 1, traces_consider.shape[1]):
                extended_traces[:, counter] = traces_consider[:, i] * traces_consider[:, j]
                counter += 1
        print("extended_traces.shape: ", extended_traces.shape)
    elif masking_order == 2:
        twice_traces = np.zeros(
            (traces_consider.shape[0], traces_consider.shape[1] * (traces_consider.shape[1] - 1) // 2))
        counter = 0
        for i in range(traces_consider.shape[1]):
            for j in range(i + 1, traces_consider.shape[1]):

                twice_traces[:, counter] = traces_consider[:, i] * traces_consider[:, j]
                counter += 1
        print("twice_traces.shape: ", twice_traces.shape)
        extended_traces = np.zeros((traces_consider.shape[0], (traces_consider.shape[1] * (traces_consider.shape[1] - 1) // 2)* traces_consider.shape[1]) )
        counter = 0
        for i in range(twice_traces.shape[1]):
            for j in range(traces_consider.shape[1]):
                extended_traces[:, counter] = twice_traces[:, i] * traces_consider[:, j]
                counter += 1
        print("extended_traces.shape: ", extended_traces.shape)
    return extended_traces

def obtain_plaintext(dataset,all_masks, correct_key):
    # print(all_masks)
    if dataset == "AES_HD_ext_plaintext":
        return None
    elif dataset == "AES_HD_ext_sbox": #[sbox_inv(pt_15^k),pt_11]
        sbox_inv = all_masks[:,0]
        getting_plain = np.zeros(all_masks.shape[0]).astype(np.uint8)
        for i in range(sbox_inv.shape[0]):
            pt_k = AES_Sbox[sbox_inv[i]]
            getting_plain[i] = pt_k^correct_key
        return getting_plain
    elif dataset == "AES_HD_ext_label":  # [sbox_inv(pt_15^k)^pt_11,pt_11]
        sbox_inv = all_masks[:, 0]^all_masks[:,1]
        getting_plain = np.zeros(all_masks.shape[0]).astype(np.uint8)
        for i in range(sbox_inv.shape[0]):
            pt_k = AES_Sbox[sbox_inv[i]]
            getting_plain[i] = pt_k ^ correct_key
        return getting_plain
    else:
        all_masks = all_masks.astype(np.uint8)
        correct_key = np.uint8(correct_key)
        getting_plain = np.zeros(all_masks.shape[0]).astype(np.uint8)
        for index in range(all_masks.shape[0]):
            for share in range(all_masks[index].shape[0]):
                if share == 0:
                    getting_plain[index] = all_masks[index, share]
                else:
                    getting_plain[index] ^= all_masks[index, share]
        for i in range(getting_plain.shape[0]):
            getting_plain[i] = AES_Sbox_inv[getting_plain[i]]
        getting_plain = getting_plain ^ correct_key
        return getting_plain






def aes_label_cpa(plt, correct_key, leakage):
    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key])
        elif leakage == 'ID':
            labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]
    return labels_for_snr
def aes_label_cpa_mask(dataset, plt, correct_key, leakage, mask, order = 1):

    num_traces = plt.shape[0]

    labels_for_snr = np.zeros(num_traces)
    if order == 0:
        for i in range(num_traces):
            if leakage == 'HW':
                labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key])
            elif leakage == 'ID':
                labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]
    elif order == 1:
        if dataset == "AES_HD_ext_plaintext":
            pt_15 = mask[:, 0]
            pt_11 = mask[:, 1]
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_snr[i] = HW(AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i])
                elif leakage == 'ID':
                    labels_for_snr[i] = AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i]
        elif dataset == "AES_HD_ext_sbox" or dataset == "AES_HD_ext_label":
            pt_15 = plt
            pt_11 = mask[:, 1]
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_snr[i] = HW(AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i])
                elif leakage == 'ID':
                    labels_for_snr[i] = AES_Sbox_inv[pt_15[i] ^ correct_key] ^ pt_11[i]
        else:
            if "simulated" in dataset:
                # mask1 = mask[0] #for original
                mask1 = mask[:, 0]
            else:
                mask1 = mask[:, 0]
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key]^mask1[i])
                elif leakage == 'ID':
                    labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]^mask1[i]
    elif order == 2:
        if "simulated" in dataset:
            # mask1 = mask[0]  #for original
            # mask2 = mask[1] #for original
            mask1 = mask[: ,0]
            mask2 = mask[: ,1]
        else:
            mask1 = mask[: ,0]
            mask2 = mask[: ,1]
        for i in range(num_traces):
            if leakage == 'HW':
                labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i])
            elif leakage == 'ID':
                labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i]
    elif order == 3:
        if "simulated" in dataset:
            # mask1 = mask[0] #for original
            # mask2 = mask[1] #for original
            # mask3 = mask[2] #for original
            mask1 = mask[:, 0]
            mask2 = mask[: ,1]
            mask3 = mask[:, 2]
        else:
            mask1 = mask[:, 0]
            mask2 = mask[: ,1]
            mask3 = mask[:, 2]
        for i in range(num_traces):
            if leakage == 'HW':
                labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i]^mask3[i])
            elif leakage == 'ID':
                labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]^mask1[i]^mask2[i]^mask3[i]
    return labels_for_snr

def aes_label_cpa_AES_HD(plt, correct_key, leakage):
    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox_inv[correct_key[15] ^ int(plt[i, 15])] ^ plt[i, 11])
        elif leakage == 'ID':
            labels_for_snr[i] = AES_Sbox_inv[correct_key[15] ^ int(plt[i, 15])] ^ plt[i, 11]
    return labels_for_snr

def cpa_method(total_samplept, number_of_traces, label, traces):
    if total_samplept == 1:
        cpa = np.zeros(total_samplept)
        cpa[0] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces])[1, 0])
    else:
        cpa = np.zeros(total_samplept)
        print("Calculate CPA")
        print(traces.shape)
        for t in tqdm(range(total_samplept)):
            cpa[t] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces, t])[1, 0])
    return cpa








def load_aes_rd(aes_rd_database_file, leakage_model='HW',train_begin = 0, train_end = 25000,test_begin = 0, test_end = 25000):
    attack_key = np.load(aes_rd_database_file + 'key.npy')[0] #only the first byte
    print(attack_key)
    X_profiling = np.load(aes_rd_database_file + 'profiling_traces_AES_RD.npy')#[:10000]
    Y_profiling = np.load(aes_rd_database_file + 'profiling_labels_AES_RD.npy')#[:10000]
    P_profiling = np.load(aes_rd_database_file + 'profiling_plaintext_AES_RD.npy')[:,0]#[:10000]
    print("X_profiling : ", X_profiling.shape)
    print("Y_profiling : ", Y_profiling.shape)
    print("P_profiling : ", P_profiling.shape)


    X_attack = np.load(aes_rd_database_file + 'attack_traces_AES_RD.npy')#[:10000]
    Y_attack = np.load(aes_rd_database_file + 'attack_plaintext_AES_RD.npy')[:,0]
    P_attack = np.load(aes_rd_database_file + 'attack_labels_AES_RD.npy')#[:,0]#[:10000]
    if leakage_model == 'HW':
        Y_profiling = np.array(calculate_HW(Y_profiling))
        Y_attack = np.array(calculate_HW(Y_attack))
    print("X_attack : ", X_attack.shape)
    print("Y_attack : ", Y_attack.shape)
    print("P_attack : ", P_attack.shape)


    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (P_profiling[train_begin:train_end],  P_attack[test_begin:test_end]), attack_key

def load_aes_hd_ext(aes_hd_database_file, leakage_model='HW',train_begin = 0, train_end = 45000,test_begin = 0, test_end = 5000):
    in_file = h5py.File(aes_hd_database_file, "r")
    last_round_key = [0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xe1, 0x3f, 0x0c, 0xc8, 0xb6, 0x63, 0x0c, 0xa6]
    X_profiling = np.array(in_file['Profiling_traces/traces'])
    C_profiling = np.array(in_file['Profiling_traces/metadata'][:]['ciphertext'], dtype='uint8')
    Y_profiling =  np.zeros(C_profiling.shape[0], dtype='uint8')
    print("Load Y_profiling")
    for i in tqdm(range(C_profiling.shape[0])):
        Y_profiling[i] = AES_Sbox_inv[last_round_key[15] ^ int(C_profiling[i, 15])] ^ C_profiling[i, 11]
    if leakage_model == 'HW':
        Y_profiling = np.array(calculate_HW(Y_profiling))

    print("X_profiling : ", X_profiling.shape)
    print("C_profiling : ", C_profiling.shape)
    print("Y_profiling : ", Y_profiling.shape)

    X_attack = np.array(in_file['Attack_traces/traces'])
    C_attack = np.array(in_file['Attack_traces/metadata'][:]['ciphertext'], dtype='uint8')
    Y_attack  = np.zeros(C_attack.shape[0], dtype='uint8')
    print("Load Y_attack ")
    for i in tqdm(range(C_attack.shape[0])):
        Y_attack[i] = AES_Sbox_inv[last_round_key[15] ^ int(C_attack[i, 15])] ^ C_attack[i, 11]
    if leakage_model == 'HW':
        Y_attack = np.array(calculate_HW(Y_attack))
    print("X_attack : ", X_attack.shape)
    print("C_attack : ", C_attack.shape)
    print("Y_attack : ", Y_attack.shape)
    attack_key = last_round_key[15]
    print("key: ", attack_key)
    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (C_profiling[train_begin:train_end],  C_attack[test_begin:test_end]), attack_key

def load_aes_hd_ext_id(aes_hd_database_file, leakage_model='HW',train_begin = 0, train_end = 45000,test_begin = 0, test_end = 5000):
    in_file = h5py.File(aes_hd_database_file, "r")
    last_round_key = [0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xe1, 0x3f, 0x0c, 0xc8, 0xb6, 0x63, 0x0c, 0xa6]
    X_profiling = np.array(in_file['Profiling_traces/traces'])
    C_profiling = np.array(in_file['Profiling_traces/metadata'][:]['ciphertext'], dtype='uint8')
    Y_profiling =  np.zeros(C_profiling.shape[0], dtype='uint8')
    print("Load Y_profiling")
    for i in tqdm(range(C_profiling.shape[0])):
        Y_profiling[i] = AES_Sbox_inv[last_round_key[15] ^ int(C_profiling[i, 15])] #^ C_profiling[i, 11]
    if leakage_model == 'HW':
        Y_profiling = np.array(calculate_HW(Y_profiling))

    print("X_profiling : ", X_profiling.shape)
    print("C_profiling : ", C_profiling.shape)
    print("Y_profiling : ", Y_profiling.shape)

    X_attack = np.array(in_file['Attack_traces/traces'])
    C_attack = np.array(in_file['Attack_traces/metadata'][:]['ciphertext'], dtype='uint8')
    Y_attack  = np.zeros(C_attack.shape[0], dtype='uint8')
    print("Load Y_attack ")
    for i in tqdm(range(C_attack.shape[0])):
        Y_attack[i] = AES_Sbox_inv[last_round_key[15] ^ int(C_attack[i, 15])] #^ C_attack[i, 11]
    if leakage_model == 'HW':
        Y_attack = np.array(calculate_HW(Y_attack))
    print("X_attack : ", X_attack.shape)
    print("C_attack : ", C_attack.shape)
    print("Y_attack : ", Y_attack.shape)
    attack_key = last_round_key[15]
    print("key: ", attack_key)
    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (C_profiling[train_begin:train_end],  C_attack[test_begin:test_end]), attack_key

def load_ascad(ascad_database_file, leakage_model='HW', byte = 2,train_begin = 0, train_end = 45000,test_begin = 0, test_end = 5000):

    in_file = h5py.File(ascad_database_file, "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'])
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))


    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, byte])
    if byte != 2:
        key_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'][:,byte])
        Y_profiling = np.zeros(P_profiling.shape[0])
        print("Loading Y_profiling")
        for i in tqdm(range(len(P_profiling))):
            Y_profiling[i] = AES_Sbox[P_profiling[i] ^ key_profiling[i]]
        if leakage_model == 'HW':
            Y_profiling = calculate_HW(Y_profiling)
    else:
        Y_profiling = np.array(in_file['Profiling_traces/labels'])  # This is only for byte 2
        if leakage_model == 'HW':
            Y_profiling = calculate_HW(Y_profiling)

    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'])
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))

    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, byte])
    attack_key = np.array(in_file['Attack_traces/metadata'][:]['key'][0, byte]) #Get the real key here (note that attack key are fixed)
    profiling_key = np.array(in_file['Profiling_traces/metadata'][:]['key'][0, byte]) #Get the real key here (note that attack key are fixed)
    print(attack_key)
    print(profiling_key)
    if byte != 2:
        print("Loading Y_attack")
        key_attack = np.array(in_file['Attack_traces/metadata'][:]['key'][:,byte])
        Y_attack = np.zeros(P_attack.shape[0])
        for i in tqdm(range(len(P_attack))):
            Y_attack[i] = AES_Sbox[P_attack[i] ^ key_attack[i]]
        if leakage_model == 'HW':
            Y_attack = calculate_HW(Y_attack)

    else:

        Y_attack = np.array(in_file['Attack_traces/labels'])
        if leakage_model == 'HW':
            Y_attack = calculate_HW(Y_attack)



    print("Information about the dataset: ")
    print("X_profiling total shape", X_profiling.shape)
    if leakage_model == 'HW':
        print("Y_profiling total shape", len(Y_profiling))
    else:
        print("Y_profiling total shape", Y_profiling.shape)
    print("P_profiling total shape", P_profiling.shape)
    print("X_attack total shape", X_attack.shape)
    if leakage_model == 'HW':
        print("Y_attack total shape", len(Y_attack))
    else:
        print("Y_attack total shape", Y_attack.shape)
    print("P_attack total shape", P_attack.shape)
    print("correct key:", attack_key)
    print()


    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (P_profiling[train_begin:train_end],  P_attack[test_begin:test_end]), attack_key

def load_ecc(ascad_database_file, leakage_model='HW', byte = 2,train_begin = 0, train_end = 45000,test_begin = 0, test_end = 5000):
    in_file = h5py.File(ascad_database_file, "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'])
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    print("X_profiling", X_profiling)
    print("X_profiling", X_profiling.shape)
    # P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, byte])
    # if byte != 2:
    #     key_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'][:,byte])
    #     Y_profiling = np.zeros(P_profiling.shape[0])
    #     print("Loading Y_profiling")
    #     for i in tqdm(range(len(P_profiling))):
    #         Y_profiling[i] = AES_Sbox[P_profiling[i] ^ key_profiling[i]]
    #     if leakage_model == 'HW':
    #         Y_profiling = calculate_HW(Y_profiling)
    # else:
    Y_profiling = np.array(in_file['Profiling_traces/label'])  # This is only for byte 2
    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)
    print("Y_profiling", Y_profiling)
    print("Y_profiling", Y_profiling.shape)
    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'])
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    print("X_attack", X_attack)
    print("X_attack", X_attack.shape)
    # P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, byte])
    # attack_key = np.array(in_file['Attack_traces/metadata'][:]['key'][0, byte]) #Get the real key here (note that attack key are fixed)
    # profiling_key = np.array(in_file['Profiling_traces/metadata'][:]['key'][0, byte]) #Get the real key here (note that attack key are fixed)
    # print(attack_key)
    # print(profiling_key)
    # if byte != 2:
    #     print("Loading Y_attack")
    #     key_attack = np.array(in_file['Attack_traces/metadata'][:]['key'][:,byte])
    #     Y_attack = np.zeros(P_attack.shape[0])
    #     for i in tqdm(range(len(P_attack))):
    #         Y_attack[i] = AES_Sbox[P_attack[i] ^ key_attack[i]]
    #     if leakage_model == 'HW':
    #         Y_attack = calculate_HW(Y_attack)
    #
    # else:

    Y_attack = np.array(in_file['Attack_traces/label'])
    if leakage_model == 'HW':
        Y_attack = calculate_HW(Y_attack)



    print("Information about the dataset: ")
    print("X_profiling total shape", X_profiling.shape)
    if leakage_model == 'HW':
        print("Y_profiling total shape", len(Y_profiling))
    else:
        print("Y_profiling total shape", Y_profiling.shape)
    print("P_profiling total shape", P_profiling.shape)
    print("X_attack total shape", X_attack.shape)
    if leakage_model == 'HW':
        print("Y_attack total shape", len(Y_attack))
    else:
        print("Y_attack total shape", Y_attack.shape)
    print("P_attack total shape", P_attack.shape)
    print("correct key:", attack_key)
    print()

    print(ok)
    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (P_profiling[train_begin:train_end],  P_attack[test_begin:test_end]), attack_key


def load_ctf(ctf_database_file, leakage_model='HW', byte = 0,train_begin = 0, train_end = 45000,test_begin = 0, test_end = 5000):

    in_file = h5py.File(ctf_database_file, "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'])
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))

    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, byte]).astype(np.uint8)
    key_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'][:,byte]).astype(np.uint8)

    Y_profiling = np.zeros(P_profiling.shape[0])
    print("Loading Y_profiling")
    for i in tqdm(range(len(P_profiling))):
        Y_profiling[i] = AES_Sbox[P_profiling[i] ^ key_profiling[i]]
    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)

    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'])
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))

    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, byte]).astype(np.uint8)
    attack_key = np.array(in_file['Attack_traces/metadata'][:]['key'][0, byte]).astype(np.uint8)#Get the real key here (note that attack key are fixed)
    profiling_key = np.array(in_file['Profiling_traces/metadata'][:]['key'][0, byte]).astype(np.uint8) #Get the real key here (note that attack key are fixed)

    print("profiling_key",profiling_key)
    print("attack_key",attack_key)
    print("Loading Y_attack")
    key_attack = np.array(in_file['Attack_traces/metadata'][:]['key'][:,byte]).astype(np.uint8)
    Y_attack = np.zeros(P_attack.shape[0])
    for i in tqdm(range(len(P_attack))):
        Y_attack[i] = AES_Sbox[P_attack[i] ^ key_attack[i]]
    if leakage_model == 'HW':
        Y_attack = calculate_HW(Y_attack)


    print("Information about the dataset: ")
    print("X_profiling total shape", X_profiling.shape)
    if leakage_model == 'HW':
        print("Y_profiling total shape", len(Y_profiling))
    else:
        print("Y_profiling total shape", Y_profiling.shape)
    print("P_profiling total shape", P_profiling.shape)
    print("X_attack total shape", X_attack.shape)
    if leakage_model == 'HW':
        print("Y_attack total shape", len(Y_attack))
    else:
        print("Y_attack total shape", Y_attack.shape)
    print("P_attack total shape", P_attack.shape)
    print("correct key:", attack_key)
    print()


    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (P_profiling[train_begin:train_end],  P_attack[test_begin:test_end]), attack_key



def load_chipwhisperer(chipwhisper_folder, leakage_model='HW',train_begin = 0,train_end = 8000, test_begin = 8000,test_end  = 10000):
    X_profiling = np.load(chipwhisper_folder + 'traces.npy')[:10000]
    Y_profiling = np.load(chipwhisper_folder + 'labels.npy')[:10000]

    if leakage_model == 'HW':
        Y_profiling = calculate_HW(Y_profiling)
        Y_profiling = np.array(Y_profiling)
    P_profiling = np.load(chipwhisper_folder + 'plain.npy')[:10000, 0]
    print("X_total : ", X_profiling.shape)
    print("C_total : ", P_profiling.shape)
    print("Y_total : ", Y_profiling.shape)
    keys = np.load(chipwhisper_folder + 'key.npy')[:10000, 0]
    return (X_profiling[train_begin:train_end], X_profiling[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_profiling[test_begin:test_end]), \
           (P_profiling[train_begin:train_end],  P_profiling[test_begin:test_end]), keys[0]

def load_chipwhisperer_desync(chipwhisper_folder,desync_lvl, leakage_model='HW'):
    chipwhisper_folder_desync = chipwhisper_folder+ 'desync_' + str(desync_lvl) + '/'

    X_profiling = np.load(chipwhisper_folder_desync + 'Profiling_traces_desync_'+str(desync_lvl)+'.npy')
    Y_profiling = np.load(chipwhisper_folder_desync + 'Profiling_label_desync_'+str(desync_lvl)+'.npy')
    P_profiling = np.load(chipwhisper_folder_desync + 'Profiling_plaintext_desync_'+str(desync_lvl)+'.npy')
    print(X_profiling)
    print("X_profiling : ", X_profiling.shape)
    print("C_profiling : ", P_profiling.shape)
    print("Y_profiling : ", Y_profiling.shape)
    X_attack = np.load(chipwhisper_folder_desync + 'Attack_traces_desync_'+str(desync_lvl)+'.npy')
    Y_attack = np.load(chipwhisper_folder_desync + 'Attack_label_desync_'+str(desync_lvl)+'.npy')
    P_attack = np.load(chipwhisper_folder_desync + 'Attack_plaintext_desync_'+str(desync_lvl)+'.npy')
    if leakage_model == 'HW':
        Y_profiling = np.array(calculate_HW(Y_profiling))
        Y_attack = np.array(calculate_HW(Y_attack))
    print("X_attack : ", X_attack.shape)
    print("C_attack : ", P_attack.shape)
    print("Y_attack : ", Y_attack.shape)

    correct_key = np.load(chipwhisper_folder_desync + 'Correct_key_desync_'+str(desync_lvl)+'.npy')
    return (X_profiling, X_attack), (Y_profiling,  Y_attack), \
           (P_profiling,  P_attack), correct_key

def load_latent_dataset(data_root, leakage_model='HW'):
    X_latent_profiling = np.load(data_root + 'X_latent_profiling.npy')
    X_latent_attack = np.load(data_root + 'X_latent_attack.npy')
    Y_profiling = np.load(data_root + 'Y_profiling.npy')
    Y_attack = np.load(data_root + 'Y_attack.npy')
    if leakage_model == 'HW':
        Y_profiling = np.array(calculate_HW(Y_profiling))
        Y_attack = np.array(calculate_HW(Y_attack))

    plt_profiling = np.load(data_root + 'plt_profiling.npy')
    plt_attack = np.load(data_root + 'plt_attack.npy')
    correct_key = np.load(data_root + 'correct_key.npy')
    print("X_latent_profiling : ", X_latent_profiling.shape)
    print("plt_profiling : ", plt_profiling.shape)
    print("Y_profiling : ", Y_profiling.shape)
    print()
    print("X_latent_attack : ", X_latent_attack.shape)
    print("plt_attack : ", plt_attack.shape)
    print("Y_attack : ", Y_attack.shape)
    return (X_latent_profiling, X_latent_attack), (Y_profiling,  Y_attack), \
           (plt_profiling,  plt_attack), correct_key



def calculate_label_guesses_chipwhisperer(traces, plt, leakage_model):
    num_traces = len(traces)
    k = np.arange(256)
    label_guesses = np.zeros((num_traces, 256))
    for i in range(num_traces):
        if leakage_model == 'HW':
            label_guesses[i] = calculate_HW(AES_Sbox[plt[i] ^ k])
        else:
            label_guesses[i] = AES_Sbox[plt[i] ^ k]
    return label_guesses


def labelize(plaintexts, keys):
    return AES_Sbox[plaintexts ^ keys]


def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]


def bit_diff( a, b):
    hw = [bin(x).count("1") for x in range(256)]
    container = np.zeros((len(a),))
    for i in range(len(a)):
        container[i] = hw[int(a[i]) ^ int(b[i])]
    return container


def calculate_MSB( data):
    if isinstance(data, (list, tuple, np.ndarray)):
        container = np.zeros((np.shape(data)), int)
        for i in range(len(data)):
            if data[i] >= 128:
                container[i] = 1
            else:
                container[i] = 0
    else:
        if data >= 128:
            container = 1
        else:
            container = 0
    return container


def calculate_LDD( k_c=34, mode='HW'):
    p = range(256)
    hw = [bin(x).count("1") for x in range(256)]
    k_all = range(256)
    container = np.zeros((len(k_all), len(p)), int)
    variance = np.zeros((256,))

    if mode == 'HW':
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] =  hw[ labelize(p[i], k_all[j])]
        for k in range(256):
            variance[k] = np.sum(
                abs(np.power(container[k_c] - container[k], 2)))

    elif mode == 'ID':
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = labelize(p[i], k_all[j])
        for k in range(256):
            variance[k] = np.sum(abs(np.power(container[k_c] - container[k], 2)))

    else:
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = calculate_MSB(labelize(p[i], k_all[j]))
        for k in range(256):
            variance[k] = np.sum(abs(container[k_c] - container[k]))
    return variance


# Objective: GE
def rk_key(rank_array, key):
    key_val = rank_array[key]
    final_rank = np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])
    if math.isnan(float(final_rank)) or math.isinf(float(final_rank)):
        return np.float32(256)
    else:
        return np.float32(final_rank)

# Compute the evolution of rank
def rank_compute(prediction, att_plt, correct_key,leakage, dataset):
    '''
    :param prediction: prediction by the neural network
    :param att_plt: attack plaintext
    :return: key_log_prob which is the log probability
    '''
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(256)
    prediction = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255)
    for i in range(nb_traces):
        for k in range(256):
            if dataset == "AES_HD_ext":
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i, AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11] ]
                else:

                    key_log_prob[k] += prediction[i, hw[AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11]] ]
            elif dataset == "AES_HD_ext_ID":
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i, AES_Sbox_inv[k ^ int(att_plt[i, 15])]]
                else:
                    key_log_prob[k] += prediction[i, hw[AES_Sbox_inv[k ^ int(att_plt[i, 15])]]]
            else:
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i,  AES_Sbox[k ^ int(att_plt[i])]]
                else:
                    key_log_prob[k] += prediction[i,  hw[ AES_Sbox[k ^ int(att_plt[i])]]]
                # print("Need to code out the this part for that dataset: in rank compute")
                # raise Exception
        rank_evol[i] =  rk_key(key_log_prob, correct_key) #this will sort it.

    return rank_evol, key_log_prob


def perform_attacks( nb_traces, predictions, plt_attack,correct_key,leakage,dataset,nb_attacks=1, shuffle=True):
    '''
    :param nb_traces: number_traces used to attack
    :param predictions: output of the neural network i.e. prob of each class
    :param plt_attack: plaintext from attack traces
    :param nb_attacks: number of attack experiments
    :param byte: byte in questions
    :param shuffle: true then it shuffle
    :return: mean of the rank for each experiments, log_probability of the output for all key
    '''
    all_rk_evol = np.zeros((nb_attacks, nb_traces)) #(num_attack, num_traces used)
    all_key_log_prob = np.zeros(256)
    for i in tqdm(range(nb_attacks)):
        if shuffle:
            l = list(zip(predictions, plt_attack)) #list of [prediction, plaintext_attack]
            random.shuffle(l) #shuffle the each other prediction
            sp, splt = list(zip(*l)) #*l = unpacking, output: shuffled predictions and shuffled plaintext.
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces] #just use the required number of traces
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]
        rank_evol, key_log_prob = rank_compute(att_pred, att_plt,correct_key,leakage=leakage,dataset=dataset)
        all_rk_evol[i] = rank_evol
        all_key_log_prob += key_log_prob

    return np.mean(all_rk_evol, axis=0), np.float32(all_key_log_prob)



def proba_to_index( proba, classes):
    number_traces = proba.shape[0]
    prediction = np.zeros((number_traces))
    for i in range(number_traces):
        sorted_index = np.argsort(proba[i])
        # Store the index of the most possible cluster
        prediction[i] = classes[sorted_index[-1]]
    return prediction

def attack_calculate_metrics(model, nb_attacks, nb_traces_attacks,correct_key, X_attack, Y_attack, plt_attack, leakage,dataset):
    # Test: Attack on the test traces
    container = np.zeros((1+256+nb_traces_attacks,))
    predictions = model.predict(X_attack[:nb_traces_attacks])
    print("predictions:",predictions.shape)
    if leakage == 'HW':
        classes = 9
    elif leakage == 'ID':
        classes = 256
    classes_labels = range(classes)
    Y_pred =  proba_to_index(predictions, classes_labels)
    accuracy = accuracy_score(Y_attack[:nb_traces_attacks], Y_pred)
    print('accuracy: ', accuracy)
    avg_rank, all_rank = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key, dataset=dataset,nb_attacks=nb_attacks, shuffle=True, leakage = leakage)

    #calculate GE
    container[257:] = avg_rank
    container[1:257] = all_rank

    # calculate accuracy

    container[0] = accuracy
    return container

def split_profiling(X_profiling, Y_profiling, plt_profiling, correct_key, dataset, percentage_val = 0.1):
    num_traces_train = int(X_profiling.shape[0] * (1-percentage_val))
    print("")
    x_train = X_profiling[:num_traces_train,:,:]
    x_val = X_profiling[num_traces_train:,:,:]
    y_train =Y_profiling[:num_traces_train]
    y_val =Y_profiling[num_traces_train:]
    plt_train = plt_profiling[:num_traces_train]
    plt_val = plt_profiling[num_traces_train:]

    keys_train = np.full(x_train.shape[0], correct_key)
    keys_val = np.full(x_val.shape[0], correct_key)
    print("Number for training:", x_train.shape[0])
    print("Number for training labels:", np.array(y_train).shape[0])
    print("Number for validation:", x_val.shape[0])
    print("Number for validation labels:", np.array(y_val).shape[0])
    return x_train,x_val, y_train,y_val, plt_train,plt_val, keys_train,keys_val
def split_attack(X_attack, Y_attack, plt_attack, percentage_val = 0.1):
    num_traces_attack = int(X_attack.shape[0] * (1-percentage_val))
    print("")
    x_explain = X_attack[:num_traces_attack,:,:]
    x_ref = X_attack[num_traces_attack:,:,:]
    y_explain =Y_attack[:num_traces_attack]
    y_ref =Y_attack[num_traces_attack:]
    plt_explain = plt_attack[:num_traces_attack]
    plt_ref = plt_attack[num_traces_attack:]


    print("Number for explain:", x_explain.shape[0])
    print("Number for ref:", x_ref.shape[0])
    return x_explain,x_ref, y_explain,y_ref, plt_explain,plt_ref
def NTGE_fn(GE):
    NTGE = float('inf')
    for i in range(GE.shape[0] - 1, -1, -1):
        if GE[i] > 0:
            break
        elif GE[i] == 0:
            NTGE = i
    return NTGE

def MSB(label):
    # print(label)
    msb = [int(j&0x1) for j in range(np.max(label)+1)]
    # print(msb)
    return [msb[i] for i in label]

def str2dict(x):
    x = str(x).replace(" ", "")
    if "{" in x:
        inputs_type2 = x.replace("{","").replace("}","")
        inputs_type1 = inputs_type2.split(",")
        z = [int(i) for i in inputs_type1]
    return z

def str2list(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type2 = x.replace("[","").replace("]","")
        inputs_type1 = inputs_type2.split(",")
        z = [int(i) for i in inputs_type1]
    return z

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def gen_trace(n_features, k = 0x03, var_noise = 0.1):
    trace = np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0, 255)
    p = random.randint(0,255)#np.random.randint(256)
    label = AES_Sbox[p ^ k]
    # print(label)
    # trace[10] = HW(label)
    for i in range(5,11):
        trace[i] = label
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    # trace = trace + np.random.normal(0, var_noise, n_features)
    trace = trace + noise_traces
    return trace, label, p


def gen_trace_mask_order1(n_features, k = 0x03, var_noise = 0.1):
    trace= np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0,255)
    p = random.randint(0,255)#np.random.randint(256)
    m = random.randint(0,255) #np.random.randint(256)
    label = AES_Sbox[p ^ k]
    # trace[10] = HW(label ^ m)
    # trace[5] = HW(m)
    for i in range(0,6):
        trace[i] = m
    for i in range(10,16):
        trace[i] = label ^ m
    # trace[10] = label ^ m
    # trace[5] = m
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    # trace = trace + np.random.normal(0, var_noise, n_features)
    trace = trace + noise_traces
    return trace, label, p, m, label ^ m

def gen_trace_mask_order2(n_features, k = 0x03, var_noise = 0.1):
    trace= np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0,255)
    p = random.randint(0,255)#np.random.randint(256)
    m1 = random.randint(0,255) #np.random.randint(256)
    m2 = random.randint(0,255) #np.random.randint(256)
    label = AES_Sbox[p ^ k]
    # trace[10] = HW(label ^ m1 ^ m2)
    # trace[5] = HW(m1)
    # trace[8] = HW(m2)
    for i in range(0,5):
        trace[i] = label ^ m1 ^ m2
    for i in range(9,15):
        trace[i] = m1
    for i in range(18,23):
        trace[i] = m2
    # trace[10] = label ^ m1 ^ m2
    # trace[5] = m1
    # trace[8] = m2
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    # trace = trace + np.random.normal(0, var_noise, n_features)
    trace = trace + noise_traces
    return trace, label, p, m1, m2, label ^ m1 ^ m2

# Leakage = SBox(p xor k) xor m1 xor m2 xor m3 + noise at sample 10
# m1, m2 and m3 at sample 5, 15 and 18
# Label = SBox(p xor k)




def gen_trace_mask_order3(n_features, k = 0x03, var_noise = 0.1):
    # trace = np.random.randint(256, size=n_features)
    trace= np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0,255)
    p = random.randint(0,255)#np.random.randint(256)
    m1 = random.randint(0,255) #np.random.randint(256)
    m2 = random.randint(0,255) #np.random.randint(256)
    m3 = random.randint(0,255) #np.random.randint(256)
    label = AES_Sbox[p ^ k]
    for i in range(0,4):
        trace[i] = label ^ m1 ^ m2 ^ m3
    for i in range(7,10):
        trace[i] = m1
    for i in range(13, 17):
        trace[i] = m2
    for i in range(18, 23):
        trace[i] = m3
    # trace[10] = label ^ m1 ^ m2 ^ m3
    # trace[5] = m1
    # trace[8] = m2
    # trace[12] = m3
    # trace[10] = HW(label ^ m1 ^ m2 ^ m3)
    # trace[5] = HW(m1)
    # trace[8] = HW(m2)
    # trace[12] = HW(m3)
    # trace = trace + np.random.normal(0, var_noise, n_features)
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    # trace = trace + np.random.normal(0, var_noise, n_features)
    trace = trace + noise_traces
    return trace, label, p, m1,m2,m3,label ^ m1 ^ m2 ^ m3



def generate_traces(n_traces,n_features= 20, order = 2):


    traces = np.zeros((n_traces, n_features))
    labels = np.zeros((n_traces,))
    masks1 = np.zeros((n_traces,))
    masks2 = np.zeros((n_traces,))
    masks3 = np.zeros((n_traces,))
    masks4 = np.zeros((n_traces,))
    plaintexts = np.zeros((n_traces))
    for i in tqdm(range(n_traces)):
        trace=None
        label=None
        p = None
        if order == 0:
            print("gen_trace_mask_order0")
            trace, label, p = gen_trace(n_features,k = 0x03, var_noise = 0.01)
            masks1[i] = label
            # print("outside trace", trace)
        elif order == 1:
            print("gen_trace_mask_order1")
            trace, label, p, m1, m2 = gen_trace_mask_order1(n_features,k = 0x03, var_noise = 0.01)
            masks1[i] = m1
            masks2[i] = m2
        elif order == 2:
            print("gen_trace_mask_order2")
            trace, label, p , m1 ,m2, m3 = gen_trace_mask_order2(n_features,k = 0x03, var_noise = 0.01)
            masks1[i] = m1
            masks2[i] = m2
            masks3[i] = m3
        elif order == 3:
            print("gen_trace_mask_order3")
            trace, label, p , m1,m2,m3, m4 = gen_trace_mask_order3(n_features,k = 0x03, var_noise = 0.01)
            masks1[i] = m1
            masks2[i] = m2
            masks3[i] = m3
            masks4[i] = m4



        traces[i, :] = trace
        labels[i] = label
        plaintexts[i] = p
    traces = traces.astype(np.int64)
    labels = labels.astype(np.int64)
    plaintexts = plaintexts.astype(np.uint8)
    masks1 = masks1.astype(np.uint8)
    masks2 = masks2.astype(np.uint8)
    masks3 = masks3.astype(np.uint8)
    masks4 = masks4.astype(np.uint8)
    masks_all = [masks1,masks2,masks3,masks4]
    return traces, labels, plaintexts, masks_all



# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def create_dataset_with_min_classes(X, Y, classes):
    sort_traces = [[] for j in range(classes)]
    for i in range(X.shape[0]):
        sort_traces[int(Y[i])].append(X[i,:])
    min_num_classes = float("inf")
    min_class = -1
    for i in range(classes):
        # print("num for class: "+ str(i), len(sort_traces[i]))
        if min_num_classes > len(sort_traces[i]):
           min_num_classes = len(sort_traces[i])
           min_class = i
    print("min_num_classes: ", min_num_classes)
    print("min_class: ", min_class)
    reduced_traces_set = np.zeros((classes*min_num_classes, X.shape[1]))
    reduced_label_set = np.zeros((classes*min_num_classes,))
    for i in range(classes):
        for trace_index in range(min_num_classes):
            reduced_traces_set[i*min_num_classes + trace_index,:] = sort_traces[i][trace_index]
            reduced_label_set[i*min_num_classes + trace_index] = i
    print("reduced_traces_set:", reduced_traces_set)
    print("reduced_traces_set:", reduced_traces_set.shape)
    print("reduced_label_set:", reduced_label_set)
    print("reduced_label_set:", reduced_label_set.shape)
    return reduced_traces_set, reduced_label_set, min_num_classes

def find_num_traces_per_classes(X, Y, classes):
    sort_traces = [[] for j in range(classes)]
    for i in range(X.shape[0]):
        sort_traces[int(Y[i])].append(X[i,:])
    num_traces_per_classes = np.zeros(classes)
    for i in range(classes):
        num_traces_per_classes[i] = len(sort_traces[i])
    return num_traces_per_classes


def print_trace(traces, path_name, image_root):
    fig_gen, ax_gen = plt.subplots(figsize=(15, 7))
    if len(traces.shape) == 3:
        traces = traces.squeeze(1)
    x_axis = [i for i in range(traces.shape[1])]
    for i, trace in enumerate(traces[:300, :]):
        ax_gen.plot(x_axis, trace)

    ax_gen.set_xlabel('Sample points', fontsize=20)
    ax_gen.set_ylabel('Voltage', fontsize=20)
    for label in (ax_gen.get_xticklabels() + ax_gen.get_yticklabels()):
        label.set_fontsize(15)
    plt.savefig(image_root + path_name)
    plt.cla()

def scale_transform_trace(original_X, new_X):
    mean_latent = np.mean(original_X, axis=0)
    std_latent = np.std(original_X, axis=0)
    print(" ------ ")
    print("mean shape: ", mean_latent.shape)
    print("mean and std original latent")
    print(mean_latent)
    print(std_latent)
    print(" ------ ")
    for x2 in range(new_X.shape[2]):
        # print(x2)
        mean_latent_new = np.mean(new_X[:, 0, x2])
        # print(mean_latent_new)
        std_latent_new = np.std(new_X[:, 0, x2])
        # print(std_latent_new)
        a_shift = std_latent[x2] / std_latent_new
        b_shift = mean_latent[x2] - a_shift * mean_latent_new
        # print(a_shift)
        # print(b_shift)
        for x1 in range(new_X.shape[0]):
            new_X[x1, 0, x2] = a_shift * new_X[x1, 0, x2] + b_shift
    print("mean and std (reconstructed) latent")
    # print(np.mean(new_X, axis=0))
    # print(np.std(new_X, axis=0))
    x_mean = np.zeros(new_X.shape[2])
    x_std = np.zeros(new_X.shape[2])
    print(x_mean.shape)
    for x1 in range(new_X.shape[2]):
        x_mean[x1] = np.mean(new_X[:, 0, x1])
        x_std[x1] = np.std(new_X[:, 0, x1])
    print(x_mean)
    print(x_std)
    print(" ----------- ")
    return new_X