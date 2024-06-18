import os

def runCmd(file, dataset):
    cmd = "python generatePSSM.py ./data/SingleLabel" + file + "/" + dataset + ".txt"
    os.system(cmd)

if __name__ == '__main__':
    datasets = ['ATbP', 'PBP', 'AAP', 'ABP', 'ACP', 'AFP', 'AHTP', 'AIP', 'AMP', 'APP', 'AVP', 'CCC', 'CPP', 'DDV',
                'QSP', 'TXP']
    files = ['test', 'training']
    for f in files:
        for data in datasets:
            runCmd(f, data)