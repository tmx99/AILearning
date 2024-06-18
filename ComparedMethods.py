from sklearn.metrics import roc_auc_score, explained_variance_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, auc
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

def calculation(probabilites, label):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(probabilites)):
        if probabilites[i] < 0.5:
            if label[i] == 1:
                tn += 1
            else:
                fn += 1
        else:
            if label[i] == 0:
                tp += 1
            else:
                fp += 1
    print("tp" + str(tp))
    print("fp" + str(fp))
    return tn, fp, fn, tp

# def ReadData(m, label):
#     correct = 0
#     probabilites = []
#     with open('testResults/CAMP_' + m + '.txt') as f:
#         for line in f:
#             probabilites.append(float(line.strip('\n').split('\t')[2]))
#     for i in range(len(probabilites)):
#         if probabilites[i] < 0.5:
#             if label[i] == 1:
#                 correct += 1
#         else:
#             if label[i] == 0:
#                 correct += 1
#     print('The ACC is ' + str(correct / 2706))
#     # import pdb
#     # pdb.set_trace()
#     tn, fp, fn, tp = calculation(probabilites, label)
#
#     label = [float(x) for x in label]
#     probabilites = [1 - float(x) for x in probabilites]
#     # print(label)
#     # print(probabilites)
#     print('The AUC is ' + str(roc_auc_score(label, probabilites)))
#
#     sn, sp, acc, mcc, p, r, f1 = calc(tn, fp, fn, tp)
#
#     lr_precision, lr_recall, _ = precision_recall_curve(label, probabilites)
#     # aupr = auc(r, p)
#     aupr = average_precision_score(label, probabilites)
#     print(auc(lr_recall, lr_precision))
#     print('sn = {}, sp = {}, acc = {}, mcc = {}'.format(sn, sp, acc, mcc))
#     print('p = {}, r = {}, f1 = {}, aupr = {}'.format(p, r, f1, aupr))
#
# def GetLabel():
#     labels = []
#     with open('dataset/SingleLabel/test/AMP_binary_test.txt') as f:
#         for line in f:
#             if line[0] == '>':
#                 labels.append(int(line.strip('\n').split('|')[1]))
#     return labels

import csv

def ReadData(dataset, label):
    correct = 0
    count = 0
    probabilites = []
    with open('testResults/Result_' + dataset + '.csv') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            if row[2] == dataset or row[2] == 'SBP':
                probabilites.append(float(row[3]))
            else:
                probabilites.append(1 - float(row[3]))
            # print(float(row[3]))
    for i in range(len(probabilites)):
        if probabilites[i] < 0.5:
            if label[i] == 1:
                correct += 1
        else:
            count += 1
            if label[i] == 0:
                correct += 1
    print(dataset)
    # print('The ACC is ' + str(correct / 2706))
    tn, fp, fn, tp = calculation(probabilites, label)

    label = [float(x) for x in label]
    probabilites = [1 - float(x) for x in probabilites]
    # print('The AUC is ' + str(roc_auc_score(label, probabilites)))

    sn, sp, acc, mcc, p, r, f1 = calc(tn, fp, fn, tp)

    lr_precision, lr_recall, _ = precision_recall_curve(label, probabilites)
    # aupr = auc(r, p)
    aupr = average_precision_score(label, probabilites)
    # print(auc(lr_recall, lr_precision))
    # print('sn = {}, sp = {}, acc = {}, mcc = {}'.format(sn, sp, acc, mcc))
    # print('p = {}, r = {}, f1 = {}, aupr = {}'.format(p, r, f1, aupr))

def GetLabel(dataset):
    labels = []
    with open('dataset/SingleLabel/test/' + dataset + '_binary_test.txt') as f:
        for line in f:
            if line[0] == '>':
                labels.append(int(line.strip('\n').split('|')[1]))
    return labels

# def ReadData(m, label):
#     correct = 0
#     probabilites = []
#     with open('testResults/ampep.tsv') as f:
#         for line in f:
#             if line[0] == 'p':
#                 continue
#             probabilites.append(float(line.strip('\n').split('\t')[1]))
#     for i in range(len(probabilites)):
#         if probabilites[i] < 0.5:
#             if label[i] == 1:
#                 correct += 1
#         else:
#             if label[i] == 0:
#                 correct += 1
#     print('The ACC is ' + str(correct / 2706))
#     # import pdb
#     # pdb.set_trace()
#     tn, fp, fn, tp = calculation(probabilites, label)
#
#     label = [float(x) for x in label]
#     probabilites = [1 - float(x) for x in probabilites]
#     # print(label)
#     # print(probabilites)
#     print('The AUC is ' + str(roc_auc_score(label, probabilites)))
#
#     sn, sp, acc, mcc, p, r, f1 = calc(tn, fp, fn, tp)
#
#     lr_precision, lr_recall, _ = precision_recall_curve(label, probabilites)
#     # aupr = auc(r, p)
#     aupr = average_precision_score(label, probabilites)
#     print(auc(lr_recall, lr_precision))
#     print('sn = {}, sp = {}, acc = {}, mcc = {}'.format(sn, sp, acc, mcc))
#     print('p = {}, r = {}, f1 = {}, aupr = {}'.format(p, r, f1, aupr))

# def GetLabel():
#     labels = []
#     with open('dataset/SingleLabel/test/AMP_binary_test.txt') as f:
#         for line in f:
#             if line[0] == '>':
#                 labels.append(int(line.strip('\n').split('|')[1]))
#     return labels

if __name__ == '__main__':
    datasets = ['AAP', 'ABP', 'ACP', 'AIP', 'AVP', 'CPP', 'QSP', 'PBP']
    # methods = ['SVM', 'RF', 'DA']
    # datasets = ['PBP']
    for data in datasets:
        label = GetLabel(data)
    # for m in methods:
        ReadData(data, label)