# -*- coding: utf-8 -*-
import sys, os, subprocess
import argparse
sys.path.append('Utility')
from multiprocessing import Pool
from Bio import SeqIO
import time

complet_n=0
def generateFasta(input_file):
    print('Generating fasta files...')
    fasta_dir = Profile_HOME+'/fasta'
    if not os.path.isdir(fasta_dir):
        os.makedirs(fasta_dir)
    prot_list = list(SeqIO.parse(input_file, 'fasta'))
    with open(Profile_HOME+'/id_list.txt','w') as f:
        for protein in prot_list:
            f.write((protein.id).split('|')[0]+'_' +(protein.id).split('|')[1]+ '\n')
            print('-------------'+(protein.id).split('|')[0]+'_' +(protein.id).split('|')[1]+ '\n')
    for protein in prot_list:
        tname = protein.name.split('|')
        if len(tname) > 1:
            name = tname[0].strip('>')+'_'+tname[1]
        else:
            name = tname[0]
        print('------------name: '+name)
        #tname = name.split('.')
        # if len(tname) > 1:
        #     name = str(tname[0])+str(tname[1])
        # else:
        #     name = tname[0]
        fasta_file = fasta_dir + '/' + name + '.fasta'
        with open(fasta_file, 'w') as wf:
            wf.write('>' + name + '\n')
            wf.write(str(protein.seq) + '\n')

def run_search(fd):
    protein_name = fd.split('.')[0]
    global complet_n
    complet_n += 1
    print('Processing:%s---%d' % (protein_name, complet_n*6))
    outfmt_type = 5
    num_iter = 3
    evalue_threshold = 0.05
    fasta_file = Profile_HOME + '/fasta/' + protein_name + '.fasta'
    xml_file = Profile_HOME + '/xml/' + protein_name + '.xml'
    pssm_file = Profile_HOME + '/pssm/' + protein_name + '.pssm'
    if os.path.isfile(pssm_file):
        pass
    else:
        cmd = ' '.join([BLAST,
                        '-query ' + fasta_file,
                        '-db ' + BLAST_DB,
                        '-out ' + xml_file,
                        '-evalue ' + str(evalue_threshold),
                        '-num_iterations ' + str(num_iter),
                        '-outfmt ' + str(outfmt_type),
                        '-out_ascii_pssm ' + pssm_file,  # Write the pssm file
                        '-num_threads ' + '6']
                       )
        return_code = subprocess.call(cmd, shell=True)

def run_blast():
    print('Generating PSSM:')
    fasta_dir = Profile_HOME + '/fasta'
    seq_DIR = os.listdir(fasta_dir)
    pssm_dir = Profile_HOME + '/pssm'
    if not os.path.isdir(pssm_dir):
        os.makedirs(pssm_dir)
    xml_dir = Profile_HOME + '/xml'
    if not os.path.isdir(xml_dir):
        os.makedirs(xml_dir)

    # for d in seq_DIR:
    #     run_simple_search(d)

    pool = Pool(8)
    results = pool.map(run_search, seq_DIR)
    pool.close()
    pool.join()

def main(args):
    file_path = args.fasta_file
    global Profile_HOME
    Profile_HOME = os.path.split(file_path)[0] + '/' + os.path.split(file_path)[1].split('.')[0]
    if not os.path.isdir(Profile_HOME):
        os.makedirs(Profile_HOME)
    generateFasta(file_path)
    run_blast()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', type=str)
    args = parser.parse_args()
    global BLAST
    global BLAST_DB
    BLAST = '/home/guoyichen/BioSeq-Analysis2.0_py3-Seq/psiblast/psiblast'
    BLAST_DB = '/home/guoyichen/BioSeq-Analysis2.0_py3-Seq/psiblast/nrdb90/nrdb90'
    print('Start!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    main(args)
    print('End!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
