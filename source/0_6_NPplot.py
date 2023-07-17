import os
import argparse
import scipy.stats as stats
import pandas as pd
import statistic
import numpy as np
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import matplotlib.pyplot as plt
import math

cdktypes = ['standard', 'extended', 'graph', 'maccs', 'pubchem', 'estate', 'hybridization', #'lingo', 
            'klekota-roth', 
            #'shortestpath', 'signature', 
            'substructure']
rdktypes = ['rdkit', 'morgan', #'rdk-maccs', 'topological-torsion', 
            'avalon']#, 'atom-pair']
babeltypes = ['fp2', 'fp4']#,fp3']

alltypes = cdktypes
alltypes.extend(rdktypes)
alltypes.extend(babeltypes)

def makePlot(listNP, listSim, sType):
        
     # draw NP histogram

    serNP = pd.Series(listNP)
    nNonZero = 0
    for np in listNP:
        if np >= 0.05:   nNonZero += 1
    print(nNonZero)
    plt.subplot(1,1,1)
    #plt.title("NP (" + sType + ")")
    plt.xlabel('NP scores')
    plt.ylabel('# of compounds')
    plt.xlim([-0.05,1.45])
    plt.ylim([0,50])
    serNP.hist(bins=30)
        
    """
    # draw edge similarity
    serSim = pd.Series(listSim)

    plt.subplot(2,1,2)
    plt.xlabel('Edge Similarity')
    plt.ylabel('Density')
    plt.xlim([0,1])
    plt.ylim([0,200000])
    serSim.hist(bins=100)
    """    
    #plt.figure(figsize=(5,10))
    plt.show()
    plt.savefig(sType+"_NP.png")
    



def analNP(sNPFile, sEdgeFile, sType):
    listTestX, listTestY = [], []
    
        
    listNP = []
    with open(sNPFile, "r") as fp:
        for sLine in fp.readlines():
            if sLine[0] == 'p':  continue
            item = sLine.split()
            dPred = float(item[1])
            listNP.append(dPred)
    
    listSim = []
    nSeed, nInter, nComp = 0,0,0
    with open(sEdgeFile, "r") as fp:
        for sLine in fp.readlines():
            item = sLine.split()
            dSim = float(item[2])
            listSim.append(dSim)

            if dSim < 0.85:     continue
            if item[0][0] == 'p' or item[0][0] == 't':
                if item[1][0] == 'p' or item[1][0] == 't':      nSeed += 1
                else:                                           nInter += 1
            else:
                if item[1][0] == 'p' or item[1][0] == 't':      nInter += 1
                else:                                           nComp += 1
    
    makePlot(listNP, listSim, sType)
    print("comp/seed/inter: ", nComp, nSeed, nInter, nComp*100/(35757*35756), nSeed*100/(297*298), nInter*100/(35757*298))
    return 

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",type=str,dest="networkFile",action="store",help="NP File")
    parser.add_argument("-f",type=str,dest="dataFile",action="store",help="Data File")
    args = parser.parse_args()
    
    for sType in alltypes:
        analNP(args.networkFile+sType+".txt", args.networkFile.replace("NP_","edge_")+sType+".txt", sType)
        break


