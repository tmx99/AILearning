def GenerateFile(dataset, datasets):
    nonSequences = []
    for data in datasets:
        if data is not dataset:
            with open('E:/ycguo/peptide2L/dataset/SingleLabel/test/' + data + '.txt') as f:
                for line in f:
                    if line[0] != '>':
                        nonSequences.append(line.strip('\n'))
    sequences = []
    with open('E:/ycguo/peptide2L/dataset/SingleLabel/test/' + dataset + '.txt') as f:
        for line in f:
            if line[0] != '>':
                sequences.append(line.strip('\n'))
    id = 1
    file_num = 1
    f = open('E:/ycguo/peptide2L/dataset/SingleLabel/test/' + dataset + '_binary_test' + str(file_num) + '.txt', 'w')
    for seq in sequences:
        f.write('>' + str(id) + '|0\n')
        id += 1
        f.write(seq + '\n')
        if id % 500 == 1:
            f.close()
            file_num += 1
            f = open('E:/ycguo/peptide2L/dataset/SingleLabel/test/' + dataset + '_binary_test' + str(file_num) + '.txt', 'w')
    for seq in nonSequences:
        f.write('>' + str(id) + '|1\n')
        id += 1
        f.write(seq + '\n')
        if id % 500 == 1:
            f.close()
            file_num += 1
            f = open('E:/ycguo/peptide2L/dataset/SingleLabel/test/' + dataset + '_binary_test' + str(file_num) + '.txt', 'w')



if __name__ == '__main__':
    datasets = ['AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'ATbP', 'AVP', 'CCC', 'CPP', 'DDV', 'PBP',
                'QSP', 'TXP']
    for dataset in datasets:
        GenerateFile(dataset, datasets)