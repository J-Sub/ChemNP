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

MAX_CUT = 5000

def MakeSeedFile(sSeedFile, dfSeed, dSeedCut):
    nNumPositive = 0
    nNumNegative = 0
    with open(sSeedFile, "w") as fp:
        dLogCut = math.log10(dSeedCut)
        dLogMaxCut = math.log10(MAX_CUT)
        for idx, row in dfSeed.iterrows():
            dPred = row['pred']
            if dPred < dSeedCut:      
                dfSeed.loc[idx, 'id'] = row['id'].replace("t", "p")
                #fp.write(dfSeed.loc[idx,'id'] + "\t" + str(round(dLogCut - math.log10(dPred),4)) + "\n")
                fp.write(dfSeed.loc[idx,'id'] + "\t" + "1" + "\n")
                nNumPositive += 1
            """ 
            elif dPred >= MAX_CUT:
                dfSeed.loc[idx, 'id'] = row['id'].replace("t", "n")
                fp.write(dfSeed.loc[idx,'id'] + "\t" + str(round(dLogMaxCut - math.log10(dPred),4)) + "\n")
                nNumNegative += 1
            """
    print("- Tested Compound(all/seed(p/n)): ", len(dfSeed), nNumPositive, nNumNegative)    
    return dfSeed


def MakeEdgeFile(sEdgeFile, dfComp, dfSeed, dictSimComp, dictSimSeed, dictSimCompSeed, dEdgeCut):
    # make edge
    listComp = dfComp['id'].tolist()
    listSeed = dfSeed['id'].tolist()
    dictComp, dictSeed = {}, {}
    for i in range(len(listComp)):  dictComp[str(i)] = str(listComp[i])
    for i in range(len(listSeed)):  dictSeed[str(i)] = str(listSeed[i])
    for sType in dictSimSeed.keys():
        sFinalEdgeFile = sEdgeFile + "_" + sType + ".txt"
        nCount = 0
        with open(sFinalEdgeFile, "w") as fp:
            # Seed
            for sLine in dictSimSeed[sType]:
                item = sLine.split("\t")
                fp.write(dictSeed[item[0]] + "\t" + dictSeed[item[1]] + "\t" + item[2] + "\n")
            # Comp
            for sLine in dictSimComp[sType]:
                item = sLine.split("\t")
                fp.write(dictComp[item[0]] + "\t" + dictComp[item[1]] + "\t" + item[2] + "\n")
            # Comp and Seed
            nCountSeed = len(dictSimSeed[sType])
            listT = []
            for i in range(101):    listT.append([])
            for sLine in dictSimCompSeed[sType]:
                item = sLine.split("\t")
                listT[int(float(item[2]) * 100)].append(dictSeed[item[0]] + "\t" + dictComp[item[1]] + "\t" + item[2] + "\n")

            for i in reversed(range(101)):
                if i >= 85:     
                    for sLine in listT[i]:  fp.write(sLine)
                    nCount += len(listT[i])
                elif i < 40 or nCount >= nCountSeed:
                    print(i+1)
                    break
                else:
                    for sLine in listT[i]:  fp.write(sLine)
                    nCount += len(listT[i])


            print(sType, "# of (Comp/Seed/combined): ", len(dictSimComp[sType]), nCountSeed, nCount)


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

    sSeedFile = sOutputPath + "seed_" + str(int(dSeedCut)) + ".txt"
    sEdgeFile = sOutputPath + "edge_" + str(int(dSeedCut)) + "_" + str(dEdgeCut)
    dfSeed = MakeSeedFile(sSeedFile, dfSeed, dSeedCut)
    MakeEdgeFile(sEdgeFile, dfComp, dfSeed, dictSimComp, dictSimSeed, dictSimCompSeed, dEdgeCut)

    


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

