from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import torch
from torchvision.transforms import transforms
from src.dataloader2 import Classifier_Dataset, ToTensor_trace_classifier
from src.net2 import create_hyperparameter_space, MLP, CNN
from src.train2 import train_classifier
from src.utils2 import perform_attacks, NTGE_fn


def train_dl_classifier(X_profiling ,Y_profiling,X_attack ,Y_attack,plt_attack, correct_key, dataset, model_type,model_root,classes,device,nb_traces_attacks,leakage, total_num_models=100, num_epochs =50, train_models = True):
    nb_attacks = 100
    dataloadertrain = Classifier_Dataset(X_profiling, Y_profiling,X_attack ,Y_attack,plt_attack,correct_key, transform=transforms.Compose([ToTensor_trace_classifier()]))
    dataloadertrain.choose_phase("train")
    dataloaderval  = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation")
    dataloadertest = deepcopy(dataloadertrain)
    dataloadertest.choose_phase("test")
    for num_models in range(total_num_models):
        if train_models == True:

            config = np.load(model_root + "model_configuration_" + str(num_models) + ".npy", allow_pickle=True).item()
            batch_size = config["batch_size"]
            num_workers = 2
            dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=num_workers),
                           # "test": torch.utils.data.DataLoader(dataloadertest, batch_size=batch_size,
                           #                                     shuffle=True, num_workers=num_workers),
                           "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
                                                              shuffle=True, num_workers=num_workers)
                           }
            dataset_sizes = {"train": len(dataloadertrain), "test": len(dataloadertest), "val": len(dataloaderval)}
            correct_key = dataloadertrain.correct_key
            X_attack = dataloadertrain.X_attack
            Y_attack = dataloadertrain.Y_attack
            plt_attack = dataloadertrain.plt_attack
            num_sample_pts = X_attack.shape[-1]

            model = train_classifier(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device)
            torch.save(model.state_dict(), model_root + "model_" + str(num_models)+ ".pth")

        else:
            config = np.load(model_root + "model_configuration_" + str(num_models) + ".npy", allow_pickle=True).item()
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            model.load_state_dict(torch.load(model_root + "model_" + str(num_models) + ".pth"))

        attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
        predictions_wo_softmax = model(attack_traces)
        predictions = F.softmax(predictions_wo_softmax, dim=1)
        predictions = predictions.cpu().detach().numpy()
        GE, key_prob = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key, dataset=dataset,
                                       nb_attacks=nb_attacks, shuffle=True, leakage=leakage)

        NTGE = NTGE_fn(GE)
        print("GE", GE)
        print("NTGE", NTGE)
        np.save(model_root + "/result_" + str(num_models), {"GE": GE, "NTGE": NTGE})