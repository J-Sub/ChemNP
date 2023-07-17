import os
import sys
import argparse
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import statistic
import matplotlib.pyplot as plt
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import pickle
import math

MAX_CUT = 5000  #when the ic50 is more than MAX_CUT, the compound is used for a negative seed

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

def MakeSeedFile(dfSeed, dSeedCut):
    nNumPositive = 0
    nNumNegative = 0
    
    dLogCut = math.log10(dSeedCut)
    dLogMaxCut = math.log10(MAX_CUT)
    for idx, row in dfSeed.iterrows():
        dPred = row['pred']
        if dPred < dSeedCut:      
            dfSeed.loc[idx, 'id'] = row['id'].replace("t", "p")
            nNumPositive += 1

    print("- Tested Compound(all/seed): ", len(dfSeed), nNumPositive)    
    return dfSeed

def makePlot(dfData, dictSim, sType):
        
    for idx, i in enumerate(dfData.columns):
        if i == 'id':   break
        # draw KNN histogram
        print(i)
        knn = dfData[i]
        listKnn = knn.to_list()
        listKnn.sort(reverse=True)
        nDup = 0
        for j in range(len(listKnn)):
            if j == 0:  nDup += 1
            else:
                if listKnn[j] == listKnn[0]:    nDup += 1
                else:                           break
        print(i, listKnn[0], nDup)

        plt.subplot(2,2,idx+1)
        #plt.title("(k="+str(i)+")")
        if idx == 2 or idx==3:      plt.xlabel('k-NN scores')
        if idx == 0 or idx==2:      plt.ylabel('# of compounds')
        plt.xlim([-0.05,1.05])
        #plt.ylim([0,25000])
        knn.hist(bins=30)
        
        """
        # draw edge similarity
        listSim = []
        for j in range(int(i)):     listSim.extend(dictSim[j])
        listSim.sort()
        print(i, listSim[0], listSim[-1], sum(listSim)/len(listSim))
        serSim = pd.Series(listSim)

        plt.subplot(2,len(dfData.columns)-1,len(dfData.columns)-1+idx+1)
        #plt.title("(k="+str(i)+")")
        plt.xlabel('Edge Similarity')
        if idx == 0:    plt.ylabel('Density')
        plt.xlim([0,1])
        plt.ylim([0,200000])
        serSim.hist(bins=100)
        """
    #plt.figure(figsize=(5,10))
    plt.show()
    plt.savefig(sType+"_KNN.png")
    




def KNN(dfComp, dfSeed):
    listSeedId = dfSeed['id'].tolist()
    listCompId = dfComp['id'].tolist()

    # make FP 
    for sType in alltypes:
        print(sType)
        listSeedFP, listCompFP = [], []
        for result in map(statistic.convertNPtoNum, statistic.makeFP(dfSeed, sType)):    listSeedFP.append(result)
        for result in map(statistic.convertNPtoNum, statistic.makeFP(dfComp, sType)):    listCompFP.append(result)
        print("DoneFP")

        rows = []
        dictSim = {}
        for i in range(520):    dictSim[i] = []
        for compIdx, fpComp in enumerate(listCompFP):
            listRank = []
            
            for seedIdx, fpSeed in enumerate(listSeedFP):
                sSeedId = listSeedId[seedIdx]
                dSim = statistic.calcTanimoto((fpComp,fpSeed))
                listRank.append(sSeedId + "\t" + str(dSim))
            listRank = sorted(listRank, key=lambda x: float(x.split('\t')[1]), reverse=True)    
            
            nPos = 0
            nNeg = 0
            setSelectIdx = (1,3,7,31)#,127,255,511)
            dictOutput = {}
            for rankIdx, sSim in enumerate(listRank):
                if rankIdx >= 520:  break
                if sSim[0] == 'p':  nPos += 1
                else:               nNeg += 1    
                dSim = float(sSim.split('\t')[1])
                dictSim[rankIdx].append(dSim)

                if rankIdx+1 in setSelectIdx:   
                    dictOutput[str(rankIdx+1)] = nPos/(nPos+nNeg)
            
            # add dataframe
            rows.append(dictOutput)
            
        dfData = pd.DataFrame.from_dict(rows, orient='columns')
        dfData['id'] = listCompId
        print(dfData)
        dfData.to_csv('KNN_'+sType+'.csv')
        
        makePlot(dfData, dictSim, sType)
        break

    return


def MakeNetwork(sInputPath, dCompCut, dEdgeCut, dSeedCut, bIsDraw):
    sOutputPath = sInputPath + str(dCompCut) + "/"
    sSimCompFile = sInputPath + "Sim_Comp_" + str(dCompCut) + ".pickle"
    sSimCompSeedFile = sInputPath + "Sim_CompSeed_" + str(dCompCut) + ".pickle"
    sSimSeedFile = sInputPath + "Sim_Seed_" + str(dCompCut) + ".pickle"

    sDataCompFile = sInputPath + "Data_Comp_" + str(dCompCut) + ".csv"
    sDataSeedFile = sInputPath + "Data_Seed_" + str(dCompCut) + ".csv"

    # check the necessary files
    if not os.path.isdir(sOutputPath):  os.mkdir(sOutputPath)
    try:
        dfComp = pd.read_csv(sDataCompFile)
        dfSeed = pd.read_csv(sDataSeedFile)
        with open(sSimCompFile, "rb") as handle:        dictSimComp = pickle.load(handle)
        with open(sSimSeedFile, "rb") as handle:        dictSimSeed = pickle.load(handle)
        with open(sSimCompSeedFile, "rb") as handle:    dictSimCompSeed = pickle.load(handle)
    except Exception as e:
        print(e)
        return True
    print("Done: Load dataset")
    dfSeed = MakeSeedFile(dfSeed, dSeedCut)
    
    
    KNN(dfComp, dfSeed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",type=str,dest="compCutoff",action="store",help="Compound Cutoff")
    parser.add_argument("-e",type=str,dest="edgeCutoff",action="store",help="Edge Cutoff")
    parser.add_argument("-i",type=str,dest="inputPath",action="store",help="Input Path")
    parser.add_argument("-s",type=str,dest="seedCutoff",action="store",help="Seed Cutoff")
    parser.add_argument("-d","--draw",action="store_true")
    args = parser.parse_args()

    #if args.anchor == None or args.simMetric == None or args.targetGeneOrCompound == None:
    #    print("Error: -a -s -t are necessary options")
    sInputPath = args.inputPath
    MakeNetwork(sInputPath+"/", float(args.compCutoff), float(args.edgeCutoff), float(args.seedCutoff), args.draw)

