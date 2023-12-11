import random

import numpy as np
from tqdm import tqdm

from src.utils import AES_Sbox_inv, AES_Sbox, rk_key
from scipy.stats import multivariate_normal

def perform_attacks_without_log( nb_traces, predictions, plt_attack,correct_key,leakage,dataset,nb_attacks=1, shuffle=True):
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
        rank_evol, key_log_prob = rank_compute_without_log(att_pred, att_plt,correct_key,leakage=leakage,dataset=dataset)
        all_rk_evol[i] = rank_evol
        all_key_log_prob += key_log_prob
    print()
    return np.mean(all_rk_evol, axis=0), np.float32(all_key_log_prob)


def rank_compute_without_log(prediction, att_plt, correct_key,leakage, dataset):
    '''
    :param prediction: prediction by the neural network
    :param att_plt: attack plaintext
    :return: key_log_prob which is the log probability
    '''
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(256)
    rank_evol = np.full(nb_traces, 255)
    for i in range(nb_traces):
        for k in range(256):
            if dataset == "AES_HD_ext":
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i, AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11] ]
                else:

                    key_log_prob[k] += prediction[i, hw[AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11]] ]
            else:
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i,  AES_Sbox[k ^ int(att_plt[i])]]
                else:
                    key_log_prob[k] += prediction[i,  hw[ AES_Sbox[k ^ int(att_plt[i])]]]
        rank_evol[i] =  rk_key(key_log_prob, correct_key) #this will sort it.

    return rank_evol, key_log_prob


def create_cov_mean_per_class(cut_out_trace_profiling,Y_profiling, num_of_features, classes):

    split_trace_classes = [[] for i in range(classes)]
    for i in range(cut_out_trace_profiling.shape[0]):
        split_trace_classes[int(Y_profiling[i])].append(cut_out_trace_profiling[i]) #normally is just this
    mean_classes = [[] for i in range(classes)]
    cov_classes = [[] for i in range(classes)]
    print("obtain mean and covariance")
    for cl in tqdm(range(classes)):
        split_trace_classes[cl] = np.array(split_trace_classes[cl])
        mean_classes[cl].append(np.mean(split_trace_classes[cl], axis=0))
        cov_classes[cl].append(np.cov(split_trace_classes[cl].T))
    print("obtain determinant")
    if num_of_features == 1:
        determinant_classes = np.zeros(classes)  # determinant = variance
        for cl in range(classes):
            determinant_classes[cl] = cov_classes[cl][0]
    else:
        determinant_classes = np.zeros(classes)
        for cl in tqdm(range(classes)):
            determinant_classes[cl] = np.linalg.det(cov_classes[cl][0])
        print("determinant: ", determinant_classes)
    return determinant_classes, mean_classes, cov_classes

def template_attack_single_trace(trace, classes, determinant_classes, mean_classes, cov_classes):
    def cal_probability_one_class(trace, determinant, mean_cl, cov_cl):
        if len(trace) == 1:
            trace_minus_mean = (trace[0] - mean_cl[0])
            hi = trace_minus_mean ** 2 / cov_cl
            return -np.log(determinant + 1e-40) - (1 / 2) * hi
        else:
            trace_minus_mean = (trace - mean_cl).reshape(trace.shape[0], 1)
            inv_cov_matrix = np.linalg.inv(cov_cl)
            hi = np.matmul(np.matmul(trace_minus_mean.T, inv_cov_matrix), trace_minus_mean)[0][0]
            return (-1 / 2) * np.log(determinant + 1e-40) - (1 / 2) * hi

    return np.array(list(
        map(lambda i: cal_probability_one_class(trace, determinant_classes[i], mean_classes[i][0], cov_classes[i][0]),
            [j for j in range(classes)])))


def template_attack(traces, classes, determinant_classes, mean_classes, cov_classes):
    log_prob = np.zeros((traces.shape[0], classes))
    for i in tqdm(range(traces.shape[0])):
        log_prob[i, :] = template_attack_single_trace(traces[i, :], classes, determinant_classes, mean_classes,
                                                      cov_classes)
    return log_prob
