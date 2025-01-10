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
    parser.add_argument('--frn', '-frn',  help='nominal value of peak frequency (Hz)', type=float, default=43.0)
    parser.add_argument('--frs', '-frs',  help='uncertainty of peak frequency (Hz)', type=float, default=1.0)
    parser.add_argument('--wrn', '-wrn',  help='nominal value of peak width (Hz)', type=float, default=5.0)
    parser.add_argument('--wrs', '-wrs',  help='uncertainty of peak width (Hz)', type=float, default=1.0)
    parser.add_argument('--Arn', '-Arn',  help='nominal value of peak amplitude relative to 50 Hz', type=float, default=2.0)
    parser.add_argument('--Ars', '-Ars',  help='uncertainty of peak amplitude relative to 50 Hz', type=float, default=0.5)
    parser.add_argument('--tE', '-tE',  help='excitatory time constant (s)', type=float, default=0.02)
    parser.add_argument('--tI', '-tI',  help='inhibitory time constant (s)', type=float, default=0.01)
    parser.add_argument('--num_sim', '-n',  help='number of simulations', type=int, default=10000000)
    parser.add_argument('--num_samp', '-p',  help='number of posterior samples', type=int, default=10000000)
    
    args2 = parser.parse_args()
    args = vars(args2)

    frn = args['frn']
    frs = args['frs']
    wrn = args['wrn']
    wrs = args['wrs']
    Arn = args['Arn']
    Ars = args['Ars']
    tE = args['tE']
    tI = args['tI']
    num_simulations = args['num_sim']
    num_samples = args['num_samp']
    
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
    inpath = currwd + "/sbi_posterior_sample.py"
    c1 = "{:s} -frn {:f} -frs {:f} -wrn {:f} -wrs {:f} -Arn {:f} -Ars {:f} -tE {:.3f} -tI {:.3f} -n {:d} -p {:d}".format(
        inpath,frn,frs,wrn,wrs,Arn,Ars,tE,tI,num_simulations,num_samples)
    jobname="sbi_posterior_sample"+"-fr={:.1f}+-{:.1f}_wr={:1f}+-{:.1f}_Ar={:.1f}+-{:.1f}-n={:d}".format(
            frn,frs,wrn,wrs,Arn,Ars,num_samples)
            
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
