from matplotlib import pyplot as plt

def ReadFile():
    sequences = []
    countNum = 0
    with open('E:/ycguo/peptide2L/dataset/negative_val.txt') as f:
        for line in f:
            if line[0] != '>':
                if len(line.strip('\n')) <= 100:
                    sequences.append(line.strip('\n'))
                    countNum += 1
    f = open('E:/ycguo/peptide2L/dataset/negative_val.txt', 'w')
    n = 1
    for sq in sequences:
        f.write('>neg' + str(n) + '|0\n')
        f.write(sq + '\n')
        n += 1

def CombileFile():
    datasets = ['AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'ATbP', 'AVP', 'CCC', 'CPP', 'DDV', 'PBP',
                'QSP', 'TXP']
    sequences = []
    countNum = 0
    for dataset in datasets:
        with open('E:/ycguo/peptide2L/dataset/SingleLabel/' + dataset + '.txt') as f:
            for line in f:
                if line[0] != '>':
                    sequences.append(line.strip('\n'))
                    countNum += 1
    print(countNum)
    f = open('E:/ycguo/peptide2L/dataset/positive.txt', 'w')
    n = 1
    for sq in sequences:
        f.write('>pos' + str(n) + '|1\n')
        f.write(sq + '\n')
        n += 1


def countLength():
    id = 0
    with open('E:/ycguo/peptide2L/dataset/negative1.txt') as f:
        for line in f:
            if line[0] != '>':
                id += 1
                if len(line.strip('\n')) == 5:
                    print(id)

if __name__ == '__main__':
    CombileFile()