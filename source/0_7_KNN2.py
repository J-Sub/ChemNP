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
import scipy.stats as stats

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

def MakeSeed(dfSeed, dSeedCut):
    nNumPositive = 0
    nNumNegative = 0
   
    dfSeed.sort_values(['pred'], inplace=True)
    dfSeed.reset_index(drop=True, inplace=True)
    for idx, row in dfSeed.iterrows():
        if idx < dSeedCut:      
            dfSeed.loc[idx, 'id'] = row['id'].replace("t", "p")
            nNumPositive += 1
        elif idx > len(dfSeed)-dSeedCut-1:
            dfSeed.loc[idx, 'id'] = row['id'].replace("t", "n")
            nNumNegative += 1


    print(dfSeed)
    print("- Tested Compound(all/pos/neg): ", len(dfSeed), nNumPositive, nNumNegative)    
    return dfSeed

def makePlot(dfData, dictSim, sType):
    print(sType, end="\t")
    for idx, i in enumerate(dfData.columns):
        if i == 'id':   break
        # draw KNN histogram
        #print(i)
        knn = dfData[i]
        listKnn = knn.to_list()
        listKnn.sort(reverse=True)
        nDup = 0
        for j in range(len(listKnn)):
            if j == 0:  nDup += 1
            else:
                if listKnn[j] == listKnn[0]:    nDup += 1
                else:                           break
        #print(i, listKnn[0], nDup)

        plt.subplot(2,len(dfData.columns)-1,idx+1)
        plt.title("(k="+str(i)+")")
        plt.xlabel('KNN Probability')
        if idx == 0:    plt.ylabel('Density')
        plt.xlim([-0.05,1.05])
        #plt.ylim([0,25000])
        knn.hist(bins=100)
        

        listTestX, listTestY = [], []
        for ii, row in dfData.iterrows():
            listTestX.append(math.log10(row['ic50']))
            listTestY.append(row[i])

        
        plt.subplot(2,len(dfData.columns)-1,len(dfData.columns)+idx)
        plt.xlabel("log10(IC50[nm])")
        plt.ylabel("NP score")
        #plt.title(sNetworkName + "(n=" + str(len(listTestX)) + ")")
        plt.scatter(listTestX, listTestY)
        #plt.show()
        plt.savefig(sType+"_valid.png")
        corr = stats.pearsonr(listTestX, listTestY)
        print(corr.statistic, end="\t")
    print()
    #plt.figure(figsize=(5,10))
    #plt.show()
    plt.savefig(sType+"_KNN.png")
    



def KNN(dfSeed, dSeedCut):
    listSeedId = dfSeed['id'].tolist()
    

    # make FP 
    for sType in alltypes:
        #print(sType)
        listSeedFP = []
        for result in map(statistic.convertNPtoNum, statistic.makeFP(dfSeed, sType)):    
            listSeedFP.append(result)
        # print(len(listSeedFP)) #!
        #print("DoneFP")

        rows = []
        dictSim = {}
        for i in range(520):    dictSim[i] = []
        #! kNN은 similarity 내림차순으로 했을 때 
        for compIdx, fpComp in enumerate(listSeedFP):
            listRank = []
            
            for seedIdx, fpSeed in enumerate(listSeedFP):
                if compIdx == seedIdx:  continue
                sSeedId = listSeedId[seedIdx]
                if sSeedId[0] == 't':   continue

                dSim = statistic.calcTanimoto((fpComp,fpSeed))
                listRank.append(sSeedId + "\t" + str(dSim))
            listRank = sorted(listRank, key=lambda x: float(x.split('\t')[1]), reverse=True)    
            # print(len(listRank))#!
            nPos = 0
            nNeg = 0
            setSelectIdx = (1,3,7,31,63)#,127,255,511)
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
        dfData['id'] = listSeedId
        dfData['ic50'] = dfSeed['pred']
        #print(dfData)
        dfData.to_csv('KNN_'+sType+'.csv')
        
        makePlot(dfData, dictSim, sType)
        

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
        #dfComp = pd.read_csv(sDataCompFile)
        dfSeed = pd.read_csv(sDataSeedFile)
        #with open(sSimCompFile, "rb") as handle:        dictSimComp = pickle.load(handle)
        with open(sSimSeedFile, "rb") as handle:        dictSimSeed = pickle.load(handle)
        #with open(sSimCompSeedFile, "rb") as handle:    dictSimCompSeed = pickle.load(handle)
    except Exception as e:
        print(e)
        return True
    print("Done: Load dataset")
    dfSeed = MakeSeed(dfSeed, dSeedCut)
    
    
    KNN(dfSeed, dSeedCut)


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

