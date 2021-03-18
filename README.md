# Screening-Li-adsorption-on-2D-metals
This database is based on the work entitled "XXX" and link.

There are 6 files:

Li_adsorption.pth.tar: the final GCN potential for screening Li adsorption site. This potential file is generated by the CGCNN code. https://github.com/txie-93/cgcnn

li_ads_2d.csv: training set information. The first column is the work function from DFT, the second column is Li adsorption energy from DFT and GCN, and the third column is the ID from the database built by Jain et al. https://pubs.acs.org/doi/10.1021/acsenergylett.9b00876.

results_high_throughput.csv: high-throughput results from the physics-driven models. The first column is the ID from the database built by Jain et al., the second column is the work function from DFT, the third and fourth column is the Li adsorption energy from equation 3 and equation 7, respectively.

best_st.zip: structures of the found high-voltage materials with adsorbed Li from DFT.

uncertainty.py: code for selecting most uncertain samples from a set of possible samples.

randomPoints.py: code for randomly sampling Li adsorption sites.
