import read_data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from CNN import NEWCNN
from torchvision.models.resnet import Bottleneck

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_load():
    pssm_dir = 'data/SingleLabel/'
    seq_train_dir = 'data/SingleLabel/finalTraining/'

    # x = torch.linspace(1, 10, 10)
    # y = torch.linspace(10, 1, 10)
    # 把数据放在数据库中
    # torch_dataset = Data.TensorDataset(x, y)
    # import pdb
    # pdb.set_trace()

    data = read_data.Input()

    train_x, train_y, train_names = data.get_pssm_varDic_2l(seq_train_dir + 'AAP.txt',
                                                            seq_train_dir + 'ABP.txt',
                                                            seq_train_dir + 'ACP.txt',
                                                            seq_train_dir + 'AFP.txt',
                                                            seq_train_dir + 'AHTP.txt',
                                                            seq_train_dir + 'AIP.txt',
                                                            seq_train_dir + 'AMP.txt',
                                                            seq_train_dir + 'APP.txt',
                                                            seq_train_dir + 'AVP.txt',
                                                            seq_train_dir + 'CCC.txt',
                                                            seq_train_dir + 'CPP.txt',
                                                            seq_train_dir + 'DDV.txt',
                                                            seq_train_dir + 'PBP.txt',
                                                            seq_train_dir + 'QSP.txt',
                                                            seq_train_dir + 'TXP.txt',
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

    train_x = torch.Tensor(train_x).cuda()
    train_x1 = train_x.reshape([train_x.shape[0],-1,50,20])
    train_x2 = train_x1.expand(train_x.shape[0],3,50,20).cuda()
    # train_x2 = torch.repeat_interleave(train_x1, repeats=3, dim=1)
    # import pdb
    # pdb.set_trace()
    # train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y).long().cuda()
    torch_dataset = Data.TensorDataset(train_x2, train_y)
    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=8,
                                          shuffle=True, num_workers=0)
    return train_loader

def data_load1(file):
    labels = []
    features = []
    datasets = ['AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'ATbP', 'AVP', 'CCC', 'CPP', 'DDV', 'PBP',
                'QSP', 'TXP']
    for i in range(len(datasets)):
        with open('/home/guoyichen/PeptideTrans/data/SingleLabel/' + file + '/' + datasets[i] + '_PC-PseAAC-General.txt') as f:
            for line in f:
                line = line.strip('\n').split(',')
                line = [float(l) for l in line]
                features.append(line)
                labels.append(i)

    
    # y_t = np.asarray(
    #     np.zeros([len(labels), 15], dtype=np.float32))
    # for i in range(len(labels)):
    #     if int(labels[i]) == 0:
    #         y_t[i][0] = 1.0
    #     elif int(labels[i]) == 1:
    #         y_t[i][1] = 1.0
    #     elif int(labels[i]) == 2:
    #         y_t[i][2] = 1.0
    #     elif int(labels[i]) == 3:
    #         y_t[i][3] = 1.0
    #     elif int(labels[i]) == 4:
    #         y_t[i][4] = 1.0
    #     elif int(labels[i]) == 5:
    #         y_t[i][5] = 1.0
    #     elif int(labels[i]) == 6:
    #         y_t[i][6] = 1.0
    #     elif int(labels[i]) == 7:
    #         y_t[i][7] = 1.0
    #     elif int(labels[i]) == 8:
    #         y_t[i][8] = 1.0
    #     elif int(labels[i]) == 9:
    #         y_t[i][9] = 1.0
    #     elif int(labels[i]) == 10:
    #         y_t[i][10] = 1.0
    #     elif int(labels[i]) == 11:
    #         y_t[i][11] = 1.0
    #     elif int(labels[i]) == 12:
    #         y_t[i][12] = 1.0
    #     elif int(labels[i]) == 13:
    #         y_t[i][13] = 1.0
    #     elif int(labels[i]) == 14:
    #         y_t[i][14] = 1.0
    # labels = y_t
    features = torch.Tensor(features).cuda()
    features1 = features.reshape([features.shape[0],-1,1,22])
    features2 = features1.expand(features.shape[0],3,1,22).cuda()
    # train_x2 = torch.repeat_interleave(train_x1, repeats=3, dim=1)
    # import pdb
    # pdb.set_trace()
    # train_x = torch.Tensor(train_x)
    labels = torch.Tensor(labels).long().cuda()
    torch_dataset = Data.TensorDataset(features2, labels)
    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1,
                                          shuffle=True, num_workers=0)
    return train_loader
    # return features, labels
 
def train_model(trainloader):
    net = torchvision.models.resnet50(num_classes = 15).cuda()
    net.avgpool = nn.AvgPool2d(1, stride = 2, padding = 0)
    
    # net = torchvision.models.vgg13(num_classes = 16).cuda()
    # net.features[4] = nn.MaxPool2d(1, stride=1, padding=0, dilation=1, ceil_mode=False).cuda()
    # net.classifier[0] = nn.Linear(in_features = 512, out_features = 4096, bias = True).cuda()
    print(net)
    import torch.optim as optim
     
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(300):  # loop over the dataset multiple times

        running_loss = 0.0
        all_loss = 0.0
        count_num = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            # import pdb
            # pdb.set_trace()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()      
            # print statistics
            running_loss += loss.item()
            all_loss += loss.item()
            count_num += 1
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        print('epoch %d, final loss: %.3f' % (epoch + 1, all_loss / float(count_num)))
        if epoch % 100 == 0:
            torch.save(net.state_dict(),'models/resnet50_class15_Adam_epoch' + str(epoch) + '.pt')
    torch.save(net.state_dict(),'models/resnet18_class50_Adam.pt')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    train_loader = data_load()
    train_model(train_loader)
    # test_loader = data_load()
    # test_all_dataset()