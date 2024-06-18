import os

def GenerateFeature(dataset):
    methods = ['Kmer', 'DP', 'DR', 'Pse-AAC-General']
    if not os.path.exists('/home/guoyichen/Peptide2L/dataset/feature/' + dataset):
        os.makedirs('/home/guoyichen/Peptide2L/dataset/feature/' + dataset)
    for method in methods:
        cmd = 'python feature.py ../Peptide2L/dataset/' + dataset + '.txt Protein -method ' \
              + method + ' -labels -1 -f csv -out ../Peptide2L/dataset/feature/' + dataset + '/' + dataset + '_' + method + '.txt'

if __name__ == '__main__':
    datasets = ['AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'ATbP', 'AVP', 'CCC', 'CPP', 'DDV', 'PBP',
                'QSP', 'TXP']
    for data in datasets:
        GenerateFeature(data)