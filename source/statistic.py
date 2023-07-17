import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import networkx as nx
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import numba as nb
from sklearn.metrics.pairwise import cosine_similarity

@nb.njit
def convertNPtoNum(NP):
    nSize = len(NP)
    nCutNum = math.ceil(nSize/64)
    arrNum = np.array([0]*nCutNum, dtype=np.uint64)

    for i in range(nCutNum):
        nSIdx = i * 64
        for j in range(64):
            nCurrIdx = nSIdx + j
            if nCurrIdx == nSize:   break
            if NP[nCurrIdx] == 1:   arrNum[i] |= (0b1 << (64-j-1))
    return arrNum

@nb.njit
def calcTanimoto(params):
    arrNum1, arrNum2 = params
    if len(arrNum1) != len(arrNum2):    return -1
    nCntInter, nCntUnion = 0, 0
    for i in range(len(arrNum1)):
        nN1, nN2 = arrNum1[i], arrNum2[i]
        nInter = (nN1 & nN2)
        nUnion = (nN1 | nN2)
        for j in range(64):
            nTest = (0b1 << j)
            if (nTest & nInter) == nTest:   nCntInter += 1
            if (nTest & nUnion) == nTest:   nCntUnion += 1
    dSim = 0
    if nCntInter != 0 and nCntUnion != 0:   dSim = nCntInter/nCntUnion
    return dSim

def makeFP(df1, sType):
    listFP = get_fingerprints(df1['smiles'], sType)
    return np.array([fp.to_numpy() for fp in listFP])


def viewCompareComp(idx, sSmiles, dictV):
    mol = Chem.MolFromSmiles(sSmiles)
    listMol = [mol]
    listKey = ["ZINC"+idx]

    for sKey in dictV.keys():
        item = dictV[sKey].split()
        if item[0] == "0":
            mol = Chem.MolFromSmiles(item[1])
            listMol.append(mol)
            listKey.append(sKey)
    print(listKey)
    img=Draw.MolsToGridImage(listMol, molsPerRow=4, subImgSize=(200,200),legends=listKey)
    img.save(idx + ".png")


def viewNetwork(center, dictV, dictE):
    graph = nx.Graph()
    listSeedNode = []

    for sV in dictV.keys():
        graph.add_node(sV)
        if sV[0] == "p" or sV[0] == "n":    listSeedNode.append(sV)
    for sE in dictE.keys():
        item = sE.split("\t")
        graph.add_edge(item[0], item[1])
    print(len(dictV), len(dictE), len(listSeedNode))
    
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True)
    nx.draw_networkx_nodes(graph, pos, nodelist=listSeedNode, node_color="#FF1744")
    nx.draw_networkx_nodes(graph, pos, nodelist=[center], node_color="#17FF11")
    plt.show()






def RemoveDup(df1, df2):
    listMol1, listMol2 = [], []
    for idx, row in df1.iterrows():     listMol1.append(Chem.MolFromSmiles(row['smiles']))
    for idx, row in df2.iterrows():     listMol2.append(Chem.MolFromSmiles(row['smiles']))
    
    df1['dup'] = 0
    df2['dup'] = 0
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    dicMol1, dicMol2 = {}, {}
    for i in range(len(listMol1)):
        mol = listMol1[i]
        sSmiles = Chem.MolToSmiles(mol)
        if sSmiles in dicMol1:  df1.loc[i,'dup'] = 1
        else:                   dicMol1[sSmiles] = 1

    for i in range(len(listMol2)):
        mol = listMol2[i]
        sSmiles = Chem.MolToSmiles(mol)
        if sSmiles in dicMol2:      df2.loc[i,'dup'] = 1
        elif sSmiles in dicMol1:    df2.loc[i,'dup'] = 1
        else:                       dicMol2[sSmiles] = 1

    df1 = df1[df1['dup'] == 0]
    df2 = df2[df2['dup'] == 0]
    df1 = df1.drop(['dup'], axis=1)
    df2 = df2.drop(['dup'], axis=1)

    return df1, df2


def CalcTanimotoForTwoFPs(listFP1, listFP2, dCutoff):
    nListFP1 = []
    nListFP2 = []
    for itemFP in listFP1:
        nFP = 0
        nNum = len(itemFP)
        for i in range(nNum):     
            if itemFP[i] == 1:    nFP |= (0b1 << (nNum-i-1))
        nListFP1.append(nFP)
    for itemFP in listFP2:
        nFP = 0
        nNum = len(itemFP)
        for i in range(nNum):     
            if itemFP[i] == 1:    nFP |= (0b1 << (nNum-i-1))
        nListFP2.append(nFP)

    listResult = []
    for i in range(len(nListFP1)):
        for j in range(len(nListFP2)):
            nIntersection = bin((nListFP1[i] & nListFP2[j])).count("1")
            nUnion = bin((nListFP1[i] | nListFP2[j])).count("1")
            dSim = 0
            if nUnion != 0:         dSim = nIntersection/nUnion
            if dSim >= dCutoff:     
                listResult.append(str(i) + "\t" + str(j) + "\t" + str(round(dSim,4)))

    return listResult

def ParallelCalcTanimotoForTwoFPs(listParam):
    listCompFP, listSeedFP, dCutoff, sType = listParam
    print(sType, len(listCompFP), len(listSeedFP), dCutoff)
    listResult = CalcTanimotoForTwoFPs(listSeedFP, listCompFP, dCutoff)
    return listResult



def ParallelCalcTanimotoForFPs(listParam):
    listFP, dCutoff, sType = listParam
    print(sType, len(listFP), dCutoff)
    listResult = CalcTanimotoForFPs(listFP, dCutoff)
    return listResult

def CalcTanimotoForFPs(listFP, dCutoff):
    nListFP = []
    for itemFP in listFP:
        nFP = 0
        nNum = len(itemFP)
        for i in range(nNum):     
            if itemFP[i] == 1:    nFP |= (0b1 << (nNum-i-1))
        nListFP.append(nFP)

    listResult = []
    nSize = len(nListFP)
    for i in range(nSize):
        for j in range(i+1, nSize):
            nIntersection = bin((nListFP[i] & nListFP[j])).count("1")
            nUnion = bin((nListFP[i] | nListFP[j])).count("1")
            dSim = 0
            if nUnion != 0:         dSim = nIntersection/nUnion
            if dSim >= dCutoff:     
                listResult.append(str(i) + "\t" + str(j) + "\t" + str(round(dSim,4)))

    return listResult

# Cosine similarity
def CalcCosineSimilarityForVectors(listVectors, dCutoff):
    nListVectors = np.array(listVectors)

    listResult = []
    nSize = len(nListVectors)
    for i in range(nSize):
        for j in range(i+1, nSize):
            dSim = cosine_similarity([nListVectors[i]], [nListVectors[j]])[0][0]
            if dSim >= dCutoff:
                listResult.append(str(i) + "\t" + str(j) + "\t" + str(round(dSim, 4)))

    return listResult

def ParallelCalcCosineSimilarityForVectors(listParam):
    listVectors, dCutoff, sType = listParam
    print(sType, len(listVectors), dCutoff)
    listResult = CalcCosineSimilarityForVectors(listVectors, dCutoff)
    return listResult

def CalcCosineSimilarityForTwoVectors(listFP1, listFP2, dCutoff):
    arrFP1 = np.array(listFP1)
    arrFP2 = np.array(listFP2)
    similarity_matrix = cosine_similarity(arrFP1, arrFP2)
    
    listResult = []
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            similarity = similarity_matrix[i][j]
            if similarity >= dCutoff:
                listResult.append(str(i) + "\t" + str(j) + "\t" + str(round(similarity, 4)))
    
    return listResult

def ParallelCalcCosineSimilarityForTwoVectors(listParam):
    listCompFP, listSeedFP, dCutoff, sType = listParam
    print(sType, len(listCompFP), len(listSeedFP), dCutoff)
    listResult = CalcCosineSimilarityForTwoVectors(listSeedFP, listCompFP, dCutoff)
    return listResult


   
#### report network status #####
def ReportNetwork(dfT, listFP, sType, sOutputPath):
    print("### Network validation ###")
    print("The size of data:", len(listFP))
    
    if sType=='mol2vec':
        # Calculate distance between all pairs
        listResult = CalcCosineSimilarityForVectors(listFP, -1)    
    else:
        # Calculate distance between all pairs
        listResult = CalcTanimotoForFPs(listFP, 0)

    listSim = []   
    list1 = []
    list2 = []
    for i in range(11): listSim.append([])

    for sResult in listResult:
        item = sResult.split("\t")
        dPred1 = float(dfT.loc[int(item[0]), "pred"])
        dPred2 = float(dfT.loc[int(item[1]), "pred"])
        dSim = float(item[2])
        dDiff = abs(math.log10(dPred1) - math.log10(dPred2))
        listSim[int(dSim/0.1)].append(dDiff)
        list1.append(dSim)
        list2.append(dDiff)
            
    corr = stats.pearsonr(list1, list2)
    print(corr[1])


    fig = plt.figure(figsize = (20, 7))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    listXtick = []
    for i in range(1,10):
        listXtick.append("<0." + str(i) + "\n(n=" + str(len(listSim[i-1])) + ")")
    listXtick.append("<1.0\n(n=" + str(len(listSim[9])) + ")")
    listXtick.append("1.0\n(n=" + str(len(listSim[10])) + ")")
    ax.set_xticklabels(listXtick)
    ax.set_title("FingerPrint: " + sType + " (pvalue=" + str(corr[1]) + ")")
    ax.set_xlabel('Tanimoto(jaccard) Similarity between two compounds')
    ax.set_ylabel('Log10(IC50) difference')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    bp = ax.boxplot(listSim)
    #plt.show()
    plt.savefig(sOutputPath + sType + ".png")
       
    





