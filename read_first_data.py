# -*- coding: utf-8 -*-
"""
Created on 2020/7/30

@author: Jun Zhang
"""
import os
import numpy as np
from Bio import SeqIO

class Input():
    def __init__(self):
# =============================================================================
        self.blosum62 = {}
        blosum_reader = open('../lib/blosum62', 'r')
        count = 0
        for line in blosum_reader:
            count = count + 1
            if count <= 7:
                continue
            line = line.strip('\r').split()
            self.blosum62[line[0]] = [float(x) for x in line[1:21]]
# =============================================================================

    def get_protein_blosum(self, protein):
        protein_lst = []
        for aa in protein.seq:
            aa = aa.upper()
            protein_lst.append(self.blosum62[aa])
        return np.asarray(protein_lst)

    def read_pssm(self, pssm_file):
        with open(pssm_file, 'r') as f:
            lines = f.readlines()
            lines = lines[3:-6]
            pro_seq = []
            mat = []
            for line in lines:
                tmp = line.strip('\n').split()
                pro_seq.append(tmp[1])
                tmp = tmp[2:22]
                mat.append(tmp)
            mat = np.array(mat)
            mat = mat.astype(float)
            return pro_seq, mat

    def pssm_var(self, fasta_path, pssm_dir):
        mats = [] #字典，按不同长度（每100）分别存储
        names = []
        if fasta_path is not None:
            # print 'Reading:', fasta_path
            seq_record = list(SeqIO.parse(fasta_path, 'fasta'))
            for prot in seq_record:
                # n = int(len(prot.seq)/100)
                n = 100
                pssm_path = os.path.join(pssm_dir, str(prot.name) + '.pssm')
                if not os.path.isfile(pssm_path):
                    pssm = self.get_protein_blosum(prot)
                else:
                    prot_seq, pssm = self.read_pssm(pssm_path)
                    
                # if mats.get(n) is None:
                #     mats[n] = []
                # if names.get(n) is None:
                #     names[n] = []
                    
                # 按每100个划分数据
                tmp1 = np.zeros([n, 20], dtype=np.float32)
                if len(prot.seq) > pssm.shape[0]:
                    seqn = pssm.shape[0]
                else:
                    seqn = len(prot.seq)
                if seqn > n:
                    seqn = n
                for j in range(seqn):
                    tmp1[j] = pssm[j]
                mats.append(tmp1)
                names.append(prot.name)
        return mats, names

    def get_pssm_varDic_2l(self, path1, pssm_dir1):
        mats1, names1 = self.pssm_var(path1, pssm_dir1)

        labels = []
        mats = []
        names = []

        mats = np.asarray(mats1)
        names = names1

        mats = np.reshape(mats, [-1, 100, 20, 1])


        return mats, names


