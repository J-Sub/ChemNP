from multiprocessing import Pool
import itertools
import os
import pickle
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import argparse
# from openbabel import pybel
# import openbabel
# from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import statistic

cdktypes = ['standard',
            'extended',
            'graph',
            'maccs',
            'pubchem',
            'estate',
            'hybridization',
            # 'lingo',
            'klekota-roth',
            # 'shortestpath', 'signature',
            'substructure']
rdktypes = ['rdkit', 'morgan',  # 'rdk-maccs', 'topological-torsion',
            'avalon']  # , 'atom-pair']
babeltypes = ['fp2', 'fp4']  # ,fp3']
# vectypes = ['mol2vec']#, 'heteroencoder']


def MakeNetwork(sCompFile, sAnchorFile, dCutoffScore, bIsDraw, sOutputPath):
    """
    Function: Make network
    """

    print(sCompFile, sAnchorFile, dCutoffScore)
    # read compounds
    dfComp = pd.read_csv(sCompFile)
    dfComp = dfComp[dfComp['pred'] >= dCutoffScore]
    dfComp = dfComp.rename(columns={'zinc_id': 'id'})

    dfSeed = pd.read_csv(sAnchorFile)
    for idx, row in dfSeed.iterrows():
        if pd.isnull(row['IC50 (nM)']):
            if not pd.isnull(row['Kd (nM)']):
                dfSeed.loc[idx, 'IC50 (nM)'] = row['Kd (nM)']
            elif not pd.isnull(row['EC50 (nM)']):
                dfSeed.loc[idx, 'IC50 (nM)'] = row['EC50 (nM)'] 
            elif not pd.isnull(row['Ki (nM)']):
                dfSeed.loc[idx, 'IC50 (nM)'] = row['Ki (nM)']

    dfSeed = dfSeed[['Ligand SMILES', 'IC50 (nM)']]
    dfSeed["id"] = ""
    dfSeed.columns = ['smiles', 'pred', "id"]
    dfSeed = dfSeed.dropna()

    # remove duplication
    print("### Remove Duplications ###")
    print("Before:", len(dfComp), len(dfSeed))
    dfSeed, dfComp = statistic.RemoveDup(dfSeed, dfComp)
    dfSeed = dfSeed.reset_index(drop=True)
    dfComp = dfComp.reset_index(drop=True)
    print("After:", len(dfComp), len(dfSeed))

    dfSeed['index'] = dfSeed.index
    dfSeed = dfSeed.astype({'index': 'str'})
    dfSeed['id'] = 't' + dfSeed['index'] + '_' + dfSeed['pred']
    dfSeed['pred'] = dfSeed['pred'].str.replace(">", "")
    dfSeed['pred'] = dfSeed['pred'].str.replace("<", "")
    dfSeed = dfSeed.astype({'pred': 'float'})
    dfSeed.drop(['index'], axis=1, inplace=True)

    # save compound data
    dfComp.to_csv(sOutputPath + "Data_Comp_"+str(dCutoffScore)+".csv")
    dfSeed.to_csv(sOutputPath + "Data_Seed_"+str(dCutoffScore)+".csv")

    alltypes = cdktypes
    alltypes.extend(rdktypes)
    alltypes.extend(babeltypes)

    # make FP for seed
    dictSeedFP = {}
    sFPSeedFile = sOutputPath + "FP_Seed_"+str(dCutoffScore)+".pickle"
    if os.path.isfile(sFPSeedFile):
        with open(sFPSeedFile, "rb") as handle:
            dictSeedFP = pickle.load(handle)
    else:
        for sType in alltypes:
            dictSeedFP[sType] = statistic.makeFP(dfSeed, sType)
        with open(sFPSeedFile, "wb") as handle:
            pickle.dump(dictSeedFP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if bIsDraw:
        for sType in alltypes:
            statistic.ReportNetwork(dfSeed, dictSeedFP[sType],
                                    sType, sOutputPath)

    # make FP for comp
    sFPCompFile = sOutputPath + "FP_Comp_"+str(dCutoffScore)+".pickle"
    dictCompFP = {}
    if os.path.isfile(sFPCompFile):
        with open(sFPCompFile, "rb") as handle:
            dictCompFP = pickle.load(handle)
    else:
        for sType in alltypes:
            dictCompFP[sType] = statistic.makeFP(dfComp, sType)
        with open(sFPCompFile, 'wb') as handle:
            pickle.dump(dictCompFP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Calc Similarity
    sSimSeedFile = sOutputPath + "Sim_Seed_"+str(dCutoffScore)+".pickle"
    sSimCompSeedFile = sOutputPath+"Sim_CompSeed_"+str(dCutoffScore)+".pickle"
    sSimCompFile = sOutputPath + "Sim_Comp_"+str(dCutoffScore)+".pickle"
    dictSeedSim = {}
    dictCompSeedSim = {}
    dictCompSim = {}
    nCore = 16
    pool = Pool(nCore)

    if os.path.isfile(sSimSeedFile):
        with open(sSimSeedFile, "rb") as handle:
            dictSeedSim = pickle.load(handle)
    else:
        listSeedFP = []
        for sType in alltypes:
            listSeedFP.append(dictSeedFP[sType])
        listResult = pool.map(statistic.ParallelCalcTanimotoForFPs,
                              zip(listSeedFP, itertools.repeat(0.85),
                                  alltypes))
        for i in range(len(alltypes)):
            sType = alltypes[i]
            dictSeedSim[sType] = listResult[i]
        with open(sSimSeedFile, "wb") as handle:
            pickle.dump(dictSeedSim, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for sType in alltypes:
        print(len(dictSeedSim[sType]), end=" ")
    print()

    if os.path.isfile(sSimCompSeedFile):
        with open(sSimCompSeedFile, "rb") as handle:
            dictCompSeedSim = pickle.load(handle)
    else:
        listCompFP = []
        listSeedFP = []
        for sType in alltypes:
            listCompFP.append(dictCompFP[sType])
            listSeedFP.append(dictSeedFP[sType])
        listResult = pool.map(statistic.ParallelCalcTanimotoForTwoFPs,
                              zip(listCompFP, listSeedFP,
                                  itertools.repeat(0.3),
                                  alltypes))
        for i in range(len(alltypes)):
            sType = alltypes[i]
            dictCompSeedSim[sType] = listResult[i]
        with open(sSimCompSeedFile, "wb") as handle:
            pickle.dump(dictCompSeedSim, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    for sType in alltypes:
        print(len(dictCompSeedSim[sType]), end=" ")
    print()

    if os.path.isfile(sSimCompFile):
        with open(sSimCompFile, "rb") as handle:
            dictCompSim = pickle.load(handle)
    else:
        listCompFP = []
        for sType in alltypes:
            listCompFP.append(dictCompFP[sType])
        listResult = pool.map(statistic.ParallelCalcTanimotoForFPs,
                              zip(listCompFP, itertools.repeat(0.85),
                                  alltypes))
        for i in range(len(alltypes)):
            sType = alltypes[i]
            dictCompSim[sType] = listResult[i]
        with open(sSimCompFile, "wb") as handle:
            pickle.dump(dictCompSim, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for sType in alltypes:
        print(len(dictCompSim[sType]), end=" ")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, dest="compoundFile", action="store",
                        help="Compounds List")
    parser.add_argument("-c", type=str, dest="cutoffScore", action="store",
                        help="Cutoff Score")
    parser.add_argument("-a", type=str, dest="anchor", action="store",
                        help="AnchorCompound")
    parser.add_argument("-o", type=str, dest="outputpath", action="store",
                        help="Output Path")
    parser.add_argument("-d", '--draw', action='store_true')
    args = parser.parse_args()

    if args.compoundFile is None or args.cutoffScore is None:
        print("Error: -t -c are necessary options")

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    
    MakeNetwork(args.compoundFile, args.anchor, float(args.cutoffScore),
                args.draw, args.outputpath+"/")
