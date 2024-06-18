import read_data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc


def test_all_dataset():
    classes = ['AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'ATbP', 'AVP', 'CCC', 'CPP', 'DDV', 'PBP',
                'QSP', 'TXP']
    net = torchvision.models.vgg13(num_classes = 16).cpu()
    net.features[4] = nn.MaxPool2d(1, stride=1, padding=0, dilation=1, ceil_mode=False).cpu()
    net.classifier[0] = nn.Linear(in_features = 1536, out_features = 4096, bias = True).cpu()
    net.avgpool = nn.AvgPool2d(1, stride = 1, padding = 0).cpu()

    # print(net)
    net.load_state_dict(torch.load('/mnt/home/guoyichen/Peptide2L/vgg13_class16_final.pt'), strict = True)
    net.eval()
    testloader = test_data_load()
    correct = 0
    total = 0
    y_scores = []
    y_true = []
    y_predicted = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            # 0-15
            _, predicted = torch.max(outputs, 1)
            y_predicted.append(np.array(predicted[0]))
    print(y_predicted)
    



def test_data_load():
    pssm_dir = 'data/SingleLabel/'
    seq_test_dir = 'data/SingleLabel/test/'

    data = read_data.Input()

    test_x,  test_names = data.get_pssm_varDic_2l(seq_test_dir + 'AAP.txt', pssm_dir)

    test_x = torch.Tensor(test_x)
    test_x1 = test_x.reshape([test_x.shape[0],-1,50,20])
    test_x2 = test_x1.expand(test_x.shape[0],3,50,20).cpu()

    test_y = torch.ones(test_x.shape[0]).long().cpu()
    torch_dataset = Data.TensorDataset(test_x2, test_y)
    test_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1,
                                          shuffle=False, num_workers=0)
    return test_loader


if __name__ == '__main__':

    test_all_dataset()
    