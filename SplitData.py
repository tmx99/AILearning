import random

def ReadFile(dataset):
    sequences = []
    with open('E:/ycguo/peptide2L/dataset/' + dataset + '.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0] == '>':
                sequences.append(lines[i] + lines[i + 1])
            i += 1
    return sequences

def subset(alist, idxs):
    '''
        用法：根据下标idxs取出列表alist的子集
        alist: list
        idxs: list
    '''
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list

def split_list(alist, group_num=4, shuffle=True, retain_left=False):
    '''
        用法：将alist切分成group个子列表，每个子列表里面有len(alist)//group个元素
        shuffle: 表示是否要随机切分列表，默认为True
        retain_left: 若将列表alist分成group_num个子列表后还要剩余，是否将剩余的元素单独作为一组
    '''

    index = list(range(len(alist)))  # 保留下标

    # 是否打乱列表
    if shuffle:
        random.shuffle(index)

    elem_num = len(alist) // group_num  # 每一个子列表所含有的元素数量
    sub_lists = {}

    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx * elem_num, (idx + 1) * elem_num
        sub_lists['set' + str(idx)] = subset(alist, index[start:end])

    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index):  # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists['set' + str(idx)] += subset(alist, index[end:])

    return sub_lists

def SplitTestdata(sequences):
    sub_lists = split_list(sequences, group_num = 10)
    return sub_lists['set0'], sub_lists['set1'] + sub_lists['set2'] + sub_lists['set3'] + sub_lists['set4'] + sub_lists['set5'] + sub_lists['set6'] + sub_lists['set7'], sub_lists['set8'] + sub_lists['set9']

def WriteFile(dataset, testData, trainData, valData):
    f = open('E:/ycguo/peptide2L/dataset/' + dataset + '_train.txt','w')
    for sq in trainData:
        f.write(sq)
        
    f = open('E:/ycguo/peptide2L/dataset/' + dataset + '_test.txt','w')
    for sq in testData:
        f.write(sq)

    f = open('E:/ycguo/peptide2L/dataset/' + dataset + '_val.txt','w')
    for sq in valData:
        f.write(sq)

# def CombineFile(dataset):
#     sequences = []
#     with open('E:/ycguo/peptide2L/dataset/SingleLabel/training/' + dataset + '.txt', 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for i in range(len(lines)):
#             if lines[i][0] == '>':
#                 sequences.append(lines[i] + lines[i + 1])
#             i += 1
#     with open('E:/ycguo/peptide2L/dataset/SingleLabel/validation/' + dataset + '.txt', 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for i in range(len(lines)):
#             if lines[i][0] == '>':
#                 sequences.append(lines[i] + lines[i + 1])
#             i += 1
#     f = open('E:/ycguo/peptide2L/dataset/SingleLabel/finalTraining/' + dataset + '.txt','w')
#     for sq in sequences:
#         f.write(sq)

if __name__ == '__main__':
    datasets = ['positive']
    for dataset in datasets:
        sequences = ReadFile(dataset)
        valData, trainData, testData = SplitTestdata(sequences)
        WriteFile(dataset, testData, trainData, valData)
        # CombineFile(dataset)