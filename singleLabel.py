def redundancy(datasets):
    for i in range(16):
        seqs = readFile(datasets[i])
        for j in range(i):
            seqs2 = readFile2(datasets[j])
            for s in seqs:
                if s in seqs2:
                    print(s)
                    seqs.remove(s)
        writeFile(datasets[i], seqs)

def readFile(dataset):
    sequences = []
    with open('D:/ycguo/peptide2L/dataset/' + dataset + '.txt') as f:
        for line in f:
            if line[0] != '>':
                sequences.append(line.strip('\n'))
    return sequences

def readFile2(dataset):
    sequences = []
    with open('D:/ycguo/peptide2L/dataset/SingleLabel/' + dataset + '.txt') as f:
        for line in f:
            if line[0] != '>':
                sequences.append(line.strip('\n'))
    return sequences

def writeFile(dataset, sequences):
    f = open('D:/ycguo/peptide2L/dataset/SingleLabel/' + dataset + '.txt','w')
    id = 1
    for sq in sequences:
        f.write('>' + dataset + str(id) + '|0\n')
        f.write(sq + '\n')
        id += 1


if __name__ == '__main__':
    datasets = ['ATbP', 'PBP', 'QSP', 'AAP', 'APP', 'CCC', 'AHTP', 'CPP', 'DDV', 'AFP', 'ACP', 'AIP', 'AVP', 'ABP', 'TXP', 'AMP']
    redundancy(datasets)