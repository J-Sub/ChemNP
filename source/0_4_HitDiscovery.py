import os
import argparse
import scipy.stats as stats
import pandas as pd
import statistic
import numpy as np
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import matplotlib.pyplot as plt
import math

def Test(sNPPath):
    # get file
    sFileName = sNPPath.split("/")[-1]
    sNPDir = sNPPath.replace(sFileName,"")
    sListAllFile = os.listdir(sNPDir)
    sListNPFile = [file for file in sListAllFile if file.startswith(sFileName)]
    for i in range(len(sListNPFile)):   sListNPFile[i] = sNPDir + sListNPFile[i]

    # get seed file
    sSeedFile = sNPPath.split("/")[0] + "/" + sNPPath.split("/")[1] + "/Data_Seed_" + sNPPath.split("/")[2] + ".csv" 
    dfSeed = pd.read_csv(sSeedFile, index_col=0)

    
    dictSelectedNetwork = {}
    dictCandidate = {}
    dictTestComp = {}
    for sFile in sListNPFile:
        dictSeed = {}
        for id in dfSeed['id']:     dictSeed[id] = 0
        listTestX, listTestY = [], []
        sNetworkName = (sFile.split("_")[-1])[0:-4]
        
        listCandidate = []
        listTestComp = []
        nSeed = 0
        with open(sFile, "r") as fp:
            for sLine in fp.readlines():
                item = sLine.split()
                dPred = float(item[1])
                #if item[0][0] == "p":   nSeed += 1
                
                #elif item[0][0] == "t":
                if item[0][0] == "p" or item[0][0] == "t":
                    #if dPred == 0:  continue
                    sPred = item[0].split("_")[1]
                    dValue = float((sPred.replace(">","")).replace("<",""))
                    listTestX.append(-math.log10(dValue))
                    listTestY.append(dPred)
                    listTestComp.append(item)

                    sTid = item[0]
                    if sTid[0] == "p":   sTid = sTid.replace('p','t')
                    if sTid in dictSeed:    dictSeed[sTid] = 1
                    else:                   print('error')

                elif float(item[1]) != 0.0:     listCandidate.append(item)
        
        for id in dfSeed['id']:     
            if dictSeed[id] == 0:
                sPred = id.split("_")[1]
                dValue = float((sPred.replace(">","")).replace("<",""))
                listTestX.append(-math.log10(dValue))
                listTestY.append(0)
        dictTestComp[sNetworkName] = listTestComp
        dictCandidate[sNetworkName] = listCandidate
        
        plt.clf()
        plt.xlabel("log10(IC50[nm])")
        plt.ylabel("NP score")
        plt.title(sNetworkName + "(n=" + str(len(listTestX)) + ")")
        plt.scatter(listTestX, listTestY)
        #plt.show()
        plt.savefig(sNetworkName+"_valid.png")
        corr = stats.pearsonr(listTestX, listTestY)
        dictSelectedNetwork[sNetworkName] = corr
        print(sNetworkName, len(listTestX), corr.statistic, corr.pvalue)

    return dictSelectedNetwork, dictCandidate, dictTestComp

   
def HitDiscovery(dictSelectedNetwork, dictCandidate):
    # select top network
    sTopNetwork = ""
    dTopPvalue = 1.0
    sSecondNetwork = ""
    dSecondPvalue = 1.0
    dfCand = pd.DataFrame(columns=dictCandidate.keys())
    listZero = []
    for i in range(len(dictCandidate.keys())):  listZero.append(0.0)
    for sKey in dictCandidate.keys():
        dPvalue = dictSelectedNetwork[sKey].pvalue
        if  dPvalue < 0.05 and len(dictCandidate[sKey]) > 0:
            dfComp = pd.DataFrame(dictCandidate[sKey], columns=['id',sKey])
            dfComp = dfComp.set_index('id')
            dfComp[sKey] = dfComp[sKey].astype(float)
            dfComp = dfComp[dfComp[sKey] >= 0.2]

            for idx, row in dfComp.iterrows():
                if idx in dfCand.index:
                    dfCand.loc[idx,sKey] = row[sKey]
                else:
                    dfCand.loc[idx] = listZero
                    dfCand.loc[idx,sKey] = row[sKey]

            #print(sKey, len(dfComp), len(dfCand))
            #print(dfCand)
    listCount = np.count_nonzero(dfCand, axis=1)
    dfCand["sum"] = dfCand.sum(axis=1)
    dfCand["count"] = listCount
    dfCand.sort_values(by=["count","sum"], inplace=True, ascending=False)

    #print(dfCand)
    return dfCand

def Report(dfCand, sDataFile, sEdgeFile):
    nNum = 0
    sSeedFile = sDataFile.replace("Data_Comp_","Data_Seed_")
    dfComp = pd.read_csv(sDataFile)
    dfSeed = pd.read_csv(sSeedFile)
    dfComp.set_index('id', inplace=True)
    dfComp.index = dfComp.index.astype(str)
    dfSeed.set_index('id', inplace=True)

    # read edge
    dictEdge = {}
    for sCol in dfCand.columns:
        sFile = sEdgeFile + sCol + ".txt"
        listEdge = []
        with open(sFile, "r") as fp:    
            for sLine in fp.readlines():    
                item = sLine.split()
                listEdge.append(item)
        dictEdge[sCol] = listEdge

    print("DONE: read edge")
    for idx, row in dfCand.iterrows():
        sCompRow = dfComp.loc[idx]
        sSmiles = sCompRow['smiles']
        sScore = sCompRow['pred']

        print(idx, row, sSmiles, sScore)
        
        for sType in dictEdge.keys():
            dictV = {}
            dictE = {}
            dictV[idx] = 1
            
            dictSeedV = {}
            for i in range(3):
                dictT = dictV.copy()
                nAddSV = 0
                for item in dictEdge[sType]:
                    if item[0] in dictT:
                        if item[1] not in dictT:
                            if (item[1][0] == "p" or item[1][0] == "n") and item[1] not in dictSeedV:   dictSeedV[item[1]] = str(i)
                            dictV[item[1]] = 1
                        dictE[item[0]+"\t"+item[1]] = 1
                    elif item[1] in dictT:
                        if item[0] not in dictT:
                            if (item[0][0] == "p" or item[0][0] == "n") and item[0] not in dictSeedV:   dictSeedV[item[0]] = str(i)
                            dictV[item[0]] = 1
                        dictE[item[0]+"\t"+item[1]] = 1
             
                print(sType, i, len(dictSeedV))
            
            listSeed = [sSmiles]
            for sKey in dictSeedV.keys():   
                sSeedSmiles = dfSeed.loc[sKey.replace("p","t"),'smiles']
                
                dictSeedV[sKey] += "\t" + sSeedSmiles 
                listSeed.append(sSeedSmiles)
            """
            listFP = get_fingerprints(listSeed, sType)
            listFP = np.array([fp.to_numpy() for fp in listFP])
            listResult = statistic.CalcTanimotoForFPs(listFP, 0.85) 
            for sLine in listResult:    print(sLine)
            """
            
            statistic.viewCompareComp(idx, sSmiles, dictSeedV)
            statistic.viewNetwork(idx, dictV, dictE)
        
        nNum += 1
        if nNum > 1:   break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",type=str,dest="networkFile",action="store",help="NP File")
    parser.add_argument("-f",type=str,dest="dataFile",action="store",help="Data File")
    args = parser.parse_args()
    
    dictSelectedNetwork, dictCandidate, dictTestComp = Test(args.networkFile)

    # for lead identification
    for sKey in dictCandidate.keys():
        listStat = dictSelectedNetwork[sKey]
        #print(sKey, len(dictCandidate[sKey]), listStat.statistic, listStat.pvalue)
    
    dfCand = HitDiscovery(dictSelectedNetwork, dictCandidate)
    sCandFile = args.dataFile.replace("Data_Comp_", "Cand_")
    dfCand.to_csv(sCandFile)


    # for seed test
    dfTest = HitDiscovery(dictSelectedNetwork, dictTestComp)
    dfTest['activity'] = dfTest.index
    for idx, row in dfTest.iterrows():
        item = row['activity'].split("_")
        sActivity = item[1]
        if sActivity[0] == ">" or sActivity[0] == "<":  sActivity = sActivity[1:]
        dfTest.loc[idx,'activity'] = sActivity
    sTestFile = args.dataFile.replace("Data_Comp_", "Test_")
    dfTest.to_csv(sTestFile)
    #Report(dfComp1, args.dataFile, args.networkFile.replace("NP_","edge_"))
    #Report(dfComp2, args.dataFile, args.networkFile.replace("NP_","edge_"))
    
