import os
import socket
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random


def runjobs():
    """
        Function to be run in a Sun Grid Engine queuing system. For testing the output, run it like
        python runjobs.py --test 1
        
    """
    
    #--------------------------------------------------------------------------
    # Test commands option
    parser = argparse.ArgumentParser()
    parser.add_argument('--tE', '-tE', help='excitatory time constant (s)', type=float, default=0.02)
    parser.add_argument('--tI', '-tI', help='inhibitory time constant (s)', type=float, default=0.01)
    # parser.add_argument('--max_coup', '-maxW', help='maximum effective coupling magnitude', type=float, default=100)
    # parser.add_argument('--max_corr', '-maxc', help='maximum correlation coefficient for E/I noise', type=float, default=1)
    # parser.add_argument('--max_Iamp', '-maxa', help='maximum ratio of I to E noise amplitude', type=float, default=2)
    parser.add_argument('--num_sim', '-n', help='number of simulations', type=int, default=10000000)
    parser.add_argument('--test', '-t', type=int, default=0)
    parser.add_argument('--cluster_', default='burg')
    parser.add_argument('--gpu', '-g', type=int, help='whether to use gpu or not', default=0)
    parser.add_argument('--mem', '-m', type=int, help='how many GB of memory to use', default=20)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    tE = args['tE']
    tI = args['tI']
    # maxW = args['max_coup']
    # maxc = args['max_corr']
    # maxa = args['max_Iamp']
    num_simulations = args['num_sim']
    
    gpu = args['gpu'] > 0
    mem = args['mem']
    
    hostname = socket.gethostname()
    if 'ax' in hostname:
        cluster = 'axon'
    else:
        cluster = str(args["cluster_"])
    
    if (args2.test):
        print ("testing commands")
    
    #--------------------------------------------------------------------------
    # Which cluster to use

    
    if platform=='darwin':
        cluster='local'
    
    currwd = os.getcwd()
    print(currwd)
    #--------------------------------------------------------------------------
    # Ofiles folder
        
    if cluster=='haba':
        path_2_package="/rigel/theory/users/thn2112/MD_SOM_Gamma/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='moto':
        path_2_package="/moto/theory/users/thn2112/MD_SOM_Gamma/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='burg':
        path_2_package="/burg/theory/users/thn2112/MD_SOM_Gamma/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='axon':
        path_2_package="/home/thn2112/MD_SOM_Gamma/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='local':
        path_2_package="/Users/tuannguyen/MD_SOM_Gamma/scripts"
        ofilesdir = path_2_package+"/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if not os.path.exists(ofilesdir):
        os.makedirs(ofilesdir)
    
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    time.sleep(0.2)
    
    #--------------------------------------------------------------------------
    # Make SBTACH
    inpath = currwd + "/sbi_sample_train.py"
    # c1 = "{:s} -tE {:.3f} -tI {:.3f} -maxW {:.1f} -maxc {:.1f} -maxa {:.1f} -n {:d}".format(
    #     inpath,tE,tI,maxW,maxc,maxa,num_simulations)
    c1 = "{:s} -tE {:.3f} -tI {:.3f} -n {:d}".format(
        inpath,tE,tI,num_simulations)
    jobname="sbi_sample_train"+"-tE={:.3f}-tI={:.3f}-n={:d}".format(
            tE,tI,num_simulations)
            
    if not args2.test:
        jobnameDir=os.path.join(ofilesdir, jobname)
        text_file=open(jobnameDir, "w");
        os.system("chmod u+x "+ jobnameDir)
        text_file.write("#!/bin/sh \n")
        if cluster=='haba' or cluster=='moto' or cluster=='burg':
            text_file.write("#SBATCH --account=theory \n")
        text_file.write("#SBATCH --job-name="+jobname+ "\n")
        if gpu:
            text_file.write("#SBATCH -t 0-11:59  \n")
            text_file.write("#SBATCH --gres=gpu\n")
        else:
            text_file.write("#SBATCH -t 3-23:59  \n")
        text_file.write("#SBATCH --mem-per-cpu={:d}gb \n".format(mem))
        text_file.write("#SBATCH -c 1 \n")
        text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
        text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
        text_file.write("python  -W ignore " + c1+" \n")
        text_file.write("echo $PATH  \n")
        text_file.write("exit 0  \n")
        text_file.close()

        if cluster=='axon':
            os.system("sbatch -p burst " +jobnameDir);
        else:
            os.system("sbatch " +jobnameDir);
    else:
        print (c1)

if __name__ == "__main__":
    runjobs()
