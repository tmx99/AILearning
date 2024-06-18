import read_data

def data_load():
    pssm_dir = 'data/SingleLabel/'
    seq_test_dir = 'data/SingleLabel/test/'

    # x = torch.linspace(1, 10, 10)
    # y = torch.linspace(10, 1, 10)
    # 把数据放在数据库中
    # torch_dataset = Data.TensorDataset(x, y)
    # import pdb
    # pdb.set_trace()

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

    test_x = torch.Tensor(test_x).cuda()
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

def test_all_dataset(testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    test_loader = data_load()
    test_all_dataset(test_loader)