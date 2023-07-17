# Preprocessing for candidate compounds with a target (get 22 FingerPrints and their similarity)

sCompoundFile='data/CLK1.csv'
sSeedFile='data/bindingDB_CLK1.csv'
dCompoundCutoff='0.7'
sPath='result/CLK1/'
########## Make FingerPrints (5 hours for 40,000 compounds) ################
time python3 source/0_1_makeFP.py -t ${sCompoundFile} -a ${sSeedFile} -c ${dCompoundCutoff} -o ${sPath} -d

########## Make Network ##############
dEdgeCutoff=$1 #'0.85'
dSeedCutoff='500'  #298(500), 100(60), 30(17), 8(8), 4(4), 2(2) #500nmol
time python3 source/0_2_makeNetwork.py -i ${sPath} -c ${dCompoundCutoff} -e ${dEdgeCutoff} -s ${dSeedCutoff}

dSeedNum='298'
#time python3 source/0_5_KNN.py -i ${sPath} -c ${dCompoundCutoff} -e ${dEdgeCutoff} -s ${dSeedCutoff}
#time python3 source/0_7_KNN2.py -i ${sPath} -c ${dCompoundCutoff} -e ${dEdgeCutoff} -s ${dSeedNum}

########## Network Propagation #############
sNPSeed=${sPath}${dCompoundCutoff}/seed_${dSeedCutoff}.txt
for sNPEdge in $(find ${sPath}${dCompoundCutoff} -name edge_${dSeedCutoff}_${dEdgeCutoff}_*);
do
  sNPresult=$(echo $sNPEdge | sed -e "s/edge/NP/g")
  time python3 source/0_3_network_analysis.py ${sNPEdge} ${sNPSeed} -o ${sNPresult} -addBidirectionEdge True
done

########## Network Analysis #############
time python3 source/0_4_HitDiscovery.py -n ${sPath}${dCompoundCutoff}/NP_${dSeedCutoff}_${dEdgeCutoff}_ -f ${sPath}Data_Comp_${dCompoundCutoff}.csv 
#time python3 source/0_6_NPplot.py -n ${sPath}${dCompoundCutoff}/NP_${dSeedCutoff}_${dEdgeCutoff}_ -f ${sPath}Data_Comp_${dCompoundCutoff}.csv 
