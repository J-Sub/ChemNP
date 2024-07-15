# ChemNP

## Preprocessing for candidate compounds with a target (Getting 14 fingerprints and their similarity)

The following preprocessing steps are performed for candidate compounds with a target (CLK1):
  
1. **Make Fingerprints**: This step involves generating 14 fingerprints for the candidate compounds using the provided compound file (`data/CLK1.csv`), seed file (`data/bindingDB_CLK1.csv`), compound cutoff (`0.7`), output path (`result/CLK1/`), and performing fingerprint generation in parallel.  

```bash

time python3 source/0_1_makeFP.py -t data/CLK1.csv -a data/bindingDB_CLK1.csv -c 0.7 -o result/CLK1/ -d

```

## Network construction

The following steps are performed for network construction:

2. **Make Network**: In this step, a network is constructed using the provided parameters:

   - Input path: ${sPath}

   - Compound cutoff: ${dCompoundCutoff}

   - Edge cutoff: ${dEdgeCutoff}

   - Seed cutoff: ${dSeedCutoff}

```bash

time python3 source/0_2_makeNetwork.py -i ${sPath} -c ${dCompoundCutoff} -e ${dEdgeCutoff} -s ${dSeedCutoff}

```

## Network propagation

The following steps are performed for network propagation:

3. **Network propagation**: is performed using the seed and edge information.

   - The seed file : `{sPath}/{dCompoundCutoff}/seed_{dSeedCutoff}.txt`

   - and the corresponding edge files are used for propagation.
  

```bash

sNPSeed=${sPath}${dCompoundCutoff}/seed_${dSeedCutoff}.txt
for sNPEdge in  $(find ${sPath}${dCompoundCutoff} -name edge_${dSeedCutoff}_${dEdgeCutoff}_*); do
    sNPresult=$(echo  $sNPEdge  |  sed  -e  "s/edge/NP/g")
    time python3 source/0_3_network_analysis.py ${sNPEdge} ${sNPSeed} -o ${sNPresult} -addBidirectionEdge True
done

```

## Network analysis

In this final step, network analysis is performed on the propagated network.

The network propagation results
`{sPath}/{dCompoundCutoff}/NP_{dSeedCutoff}_{dEdgeCutoff}`

and the compound file
`{sPath}/Data_Comp_{dCompoundCutoff}.csv`

are utilized.
  

```bash

time python3 source/0_4_HitDiscovery.py -n ${sPath}${dCompoundCutoff}/NP_${dSeedCutoff}_${dEdgeCutoff}_ -f ${sPath}Data_Comp_${dCompoundCutoff}.csv`

```

## To run ChemNP

```
./exe.sh 0.85
```
