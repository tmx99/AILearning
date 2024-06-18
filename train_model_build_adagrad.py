import read_data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_load():
    pssm_dir = 'data/SingleLabel/'
    seq_train_dir = 'data/SingleLabel/finalTraining/'
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
    train_y = torch.Tensor(train_y).long().cuda()
    torch_dataset = Data.TensorDataset(train_x2, train_y)
    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=8,
                                          shuffle=True, num_workers=0)
    return train_loader

 
def train_model(trainloader):
    net = torchvision.models.alexnet(num_classes = 15).cuda()
    net.features[2] = nn.MaxPool2d(2, stride=1, padding=1, dilation=1, ceil_mode=False).cuda()
    # net.features[5] = nn.MaxPool2d(1, stride=1, padding=0, dilation=1, ceil_mode=False).cuda()
    # net.classifier[0] = nn.Linear(in_features = 1536, out_features = 4096, bias = True).cuda()
    print(net)
    # import torch.optim as optim
     
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        all_loss = 0.0
        count_num = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
    
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
            torch.save(net.state_dict(),'models/alexnet_class15_Adam_epoch' + str(epoch) + '.pt')
    torch.save(net.state_dict(),'models/alexnet_class15_Adam_final.pt')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader = data_load()
    train_model(train_loader)
