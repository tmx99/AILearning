def ReadFile():
    sequences = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    w = True
    count = 0
    with open('E:/ycguo/peptide2L/dataset/uniprot.fasta', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == lines[0]:
                temp = ""
                continue
            if line[0] == '>':
                count += 1
                for a in temp:
                    if a not in alphabet:
                        w = False
                        break
                if w:
                    sequences.append(temp)
                w = True
                temp = ""
            else:
                temp += line.strip('\n')
    print(count)
    return sequences

def ReadOriginFile():
    s = 0
    l = 0
    datasets = ['AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'ATbP', 'AVP', 'CCC', 'CPP', 'DDV', 'PBP',
                'QSP', 'TXP']
    for data in datasets:
        with open('E:/ycguo/peptide2L/dataset/' + data + '.txt') as f:
            for line in f:
                if line[0] != '>':
                    if s == 0 and l == 0:
                        s = len(line.strip('\n'))
                        l = len(line.strip('\n'))
                    else:
                        s  = min(s, len(line.strip('\n')))
                        l = max(l, len(line.strip('\n')))
    return s, l


def WriteFile(sequences, s, l):
    id = 0
    f = open('E:/ycguo/peptide2L/dataset/negative.txt', 'w')
    for sq in sequences:
        if len(sq) >= s and len(sq) <= l:
            f.write('>' + str(id) + '|1\n')
            f.write(sq + '\n')
            id += 1

if __name__ == '__main__':
    s, l = ReadOriginFile()
    sequences = ReadFile()
    WriteFile(sequences, s, l)
