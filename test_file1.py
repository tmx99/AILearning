import read_data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
# from data_combine_loader import dataload_singlepip
# from data_combine_loader import dataload_PSSMDT

def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    # Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    # F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1 = 2 * p * r / (p + r)
    return SN, SP, ACC, MCC, p, r, f1

def calculation(i, y_true, label):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    # import pdb
    # pdb.set_trace()
    for j in range(len(y_true)):
        if y_true[j] == i:
            if label[j] == 1.0:
                tp += 1
            else:
                fp += 1
        else:
            if label[j] == 0.0:
                tn += 1
            else:
                fn += 1
    print('tp' + str(tp))
    print('fp' + str(fp))
    return tn, fp, fn, tp

def test_all_dataset():
    classes = ['AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'ATbP', 'AVP', 'CCC', 'CPP', 'DDV', 'PBP',
                'QSP', 'TXP']
    # classes = ['AAP', 'ABP', 'ACP', 'AIP', 'AVP', 'CPP', 'PBP', 'QSP']
    net = torchvision.models.vgg13(num_classes = 16).cuda()
    # net.classifier[0] = nn.Linear(in_features = 512, out_features = 4096, bias = True).cuda()
    net.features[4] = nn.MaxPool2d(1, stride=1, padding=0, dilation=1, ceil_mode=False).cuda()
    net.classifier[0] = nn.Linear(in_features = 1536, out_features = 4096, bias = True).cuda()
    # net = torchvision.models.resnet50(num_classes = 15).cuda()
    # net.avgpool = nn.AvgPool2d(1, stride = 2, padding = 0)
    # net = torchvision.models.resnet18(num_classes = 16).cuda()
    net.avgpool = nn.AvgPool2d(1, stride = 1, padding = 0).cuda()

    # print(net)
    net.load_state_dict(torch.load('/home/guoyichen/PeptideTrans/models/vgg13_class16_final.pt'), strict = True)
    net.eval()
    testloader = test_data_load()
    correct = 0
    total = 0
    y_scores = []
    y_true = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            # import pdb
            # pdb.set_trace()
            # print(torch.sum(outputs, dim = 1))
            normal_outputs = torch.nn.functional.normalize(outputs.abs(), p = 1, dim = 1)
            normal_outputs = (1 - normal_outputs) / 15
            
            y_scores.append(normal_outputs[0].cpu())
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(predicted.cpu())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(correct / total)
    # print('Accuracy of the network on the test datasets: %d %%' % (
    #     100 * correct / total))
    batch_size = 1
    class_correct = list(0. for i in range(16))
    class_total = list(0. for i in range(16))
    roc_labels = []
    for i in range(len(classes)):
        roc_label = []
        for data in testloader:
            images, labels = data
            for l in labels:
                if i == l.item():
                    roc_label.append(1.0)
                else:
                    roc_label.append(0.0)
        roc_labels.append(roc_label)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            # c = (predicted == labels).squeeze()
            c = (predicted == labels)
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(16):
        # import pdb
        # pdb.set_trace()
        if classes[i] == 'ATbP':
            continue
        y_score = [y[i] for y in y_scores]
        print(classes[i])
        # print('AUC = {}'.format(roc_auc_score(roc_labels[i], y_score)))
        # print('Accuracy of %5s : %2d %%' % (
        #     classes[i], 100 * class_correct[i] / class_total[i]))
        lr_precision, lr_recall, _ = precision_recall_curve(roc_labels[i], y_score)
        tn, fp, fn, tp = calculation(i, y_true, roc_labels[i])
        sn, sp, acc, mcc, p, r, f1 = calc(tn, fp, fn, tp)
        aupr = auc(lr_recall, lr_precision)
        # aupr = average_precision_score(roc_labels[i], y_true)
        # print('sn = {}, sp = {}, acc = {}, mcc = {}'.format(sn, sp, acc, mcc))
        # print('p = {}, r = {}, f1 = {}, aupr = {}'.format(p, r, f1, aupr))


def test_data_load():
    pssm_dir = 'data/SingleLabel/'
    seq_test_dir = 'data/SingleLabel/test/'

    data = read_data.Input()

    test_x, test_y, test_names = data.get_pssm_varDic_2l(seq_test_dir + 'AAP.txt',
                                                        seq_test_dir + 'ABP.txt',
                                                        seq_test_dir + 'ACP.txt',
                                                        seq_test_dir + 'AFP.txt',
                                                        seq_test_dir + 'AHTP.txt',
                                                        seq_test_dir + 'AIP.txt',
                                                        seq_test_dir + 'AMP.txt',
                                                        seq_test_dir + 'APP.txt',
                                                        seq_test_dir + 'ATbP.txt',
                                                        seq_test_dir + 'AVP.txt',
                                                        seq_test_dir + 'CCC.txt',
                                                        seq_test_dir + 'CPP.txt',
                                                        seq_test_dir + 'DDV.txt',
                                                        seq_test_dir + 'PBP.txt',
                                                        seq_test_dir + 'QSP.txt',
                                                        seq_test_dir + 'TXP.txt',
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir,
                                                        pssm_dir)

    test_x = torch.Tensor(test_x)
    test_x1 = test_x.reshape([test_x.shape[0],-1,50,20])
    test_x2 = test_x1.expand(test_x.shape[0],3,50,20).cuda()
    # train_x2 = torch.repeat_interleave(train_x1, repeats=3, dim=1)
    # import pdb
    # pdb.set_trace()
    # train_x = torch.Tensor(train_x)
    test_y = torch.Tensor(test_y).long().cuda()
    torch_dataset = Data.TensorDataset(test_x2, test_y)
    test_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1,
                                          shuffle=False, num_workers=0)
    return test_loader


if __name__ == '__main__':

    test_all_dataset()
    