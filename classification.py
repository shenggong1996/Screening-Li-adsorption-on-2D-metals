import os
from math import *
from pymatgen import Structure
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline

import numpy as np
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict

def InitializeDF(source,csv):
    f = open('%s'%(csv))
    length = len(f.readlines())
    f.close()
    f = open('%s'%(csv))
    material_id = []
    formula = []
    structure = []
    bots = []
    ups = []
    for i in range(length):
        _, ID, form, bot, up = f.readline().split(',')
#        print (nat,ID)
        ID = ID.replace('\n','')
#        print (ID)
        if i == 0 or (i>=3845 and i <=3850):
            continue
#        print (ID,bot)
        material_id.append(ID)
        if (not bot.strip())  and (not up.strip()):
            bot = float(0); up = float(0)
#        Ecp.append(float(Eads)+float(wf)-5.39)
#        print (ID,bot)
        bots.append(float(bot)); ups.append(float(up))
        struct = Structure.from_file('%s/%s.cif'%(source, ID))
        structure.append(struct)
        formula.append(struct.composition.reduced_formula)
#        print (struct.composition.reduced_formula)
#        print (struct.pretty_formula)
    data = {}
    data['material_id'] = material_id
    data['formula'] = formula
    data['structure'] = structure
    data['bot'] = bots
    data['up'] = ups
    df = pd.DataFrame(data)
    return (df)

def AddFeatures(df):
    from matminer.featurizers.conversions import StrToComposition
    df = StrToComposition().featurize_dataframe(df, "formula")

    from matminer.featurizers.composition import ElementProperty

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer

    from matminer.featurizers.conversions import CompositionToOxidComposition
    from matminer.featurizers.composition import OxidationStates

    df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

    os_feat = OxidationStates()
    df = os_feat.featurize_dataframe(df, "composition_oxid")

    from matminer.featurizers.composition import ElectronAffinity

    ea_feat = ElectronAffinity()
    df = ea_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)
    
    from matminer.featurizers.composition import BandCenter

    bc_feat = BandCenter()
    df = bc_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import CohesiveEnergy

    ce_feat = CohesiveEnergy()
    df = ce_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import Miedema

    m_feat = Miedema()
    df = m_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import TMetalFraction

    tmf_feat = TMetalFraction()
    df = tmf_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import ValenceOrbital

    vo_feat = ValenceOrbital()
    df = vo_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)
 
    from matminer.featurizers.composition import YangSolidSolution

    yss_feat = YangSolidSolution()
    df = yss_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.structure import GlobalSymmetryFeatures

    gsf_feat = GlobalSymmetryFeatures()
    df = gsf_feat.featurize_dataframe(df, "structure")

    from matminer.featurizers.structure import StructuralComplexity
    sc_feat = StructuralComplexity()
    df = sc_feat.featurize_dataframe(df, "structure")
    
    from matminer.featurizers.structure import ChemicalOrdering
    co_feat = ChemicalOrdering()
    df = co_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import GlobalInstabilityIndex
    gii_feat = GlobalInstabilityIndex()
    df = gii_feat.featurize_dataframe(df, "structure",ignore_errors=True)
    
    from matminer.featurizers.structure import MaximumPackingEfficiency
    mpe_feat = MaximumPackingEfficiency()
    df = mpe_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import MinimumRelativeDistances
    mrd_feat = MinimumRelativeDistances()
#    df = mrd_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import StructuralHeterogeneity
    sh_feat = StructuralHeterogeneity()
    df = sh_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import SiteStatsFingerprint

    from matminer.featurizers.site import AverageBondLength
    from pymatgen.analysis.local_env import CrystalNN
    bl_feat = SiteStatsFingerprint(AverageBondLength(CrystalNN(search_cutoff=20)))
    df = bl_feat.featurize_dataframe(df, "structure",ignore_errors=True)
    
    from matminer.featurizers.site import AverageBondAngle
    ba_feat = SiteStatsFingerprint(AverageBondAngle(CrystalNN(search_cutoff=20)))
    df = ba_feat.featurize_dataframe(df, "structure",ignore_errors=True)
    
    from matminer.featurizers.site import BondOrientationalParameter
    bop_feat = SiteStatsFingerprint(BondOrientationalParameter())
    df = bop_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.site import CoordinationNumber
    cn_feat = SiteStatsFingerprint(CoordinationNumber())
    df = cn_feat.featurize_dataframe(df, "structure")

    return (df)


#df = InitializeDF('mp_li_compounds','window_sum.csv')

#df = AddFeatures(df)

#df = df.fillna(0)

#df.to_csv('input_classification.csv')

df = pd.read_csv('input_classification.csv')

df = df.select_dtypes(include='float64')

df = df.fillna(0)

#y = df[['K_VRH']]
y = df[['bot']]
#y = np.log10(y)
#print (y.shape)

excluded = ["bot",'up']
X = df.drop(excluded, axis=1)
#print (X.dtypes)
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))

#X = (X - X.min())/(X.max() - X.min())

#X = X.fillna(0)
#model = MT_Net(feature_len = X.shape[1], hidden_len = X.shape[1], output_len = y.shape[1], n_hidden = n_hidden)

model = RandomForestRegressor(n_estimators=50, criterion = 'mae', random_state=1,verbose=True,n_jobs=2)

#model.fit(X,y)

#crossvalidation = KFold(n_splits=179, shuffle=True, random_state=1)
#scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=1)
#mae_scores = [s for s in scores]
#r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)

#print('Cross-validation results:')
#print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
#print('Folds: %i, mean MAE: %.3f' % (len(scores), np.mean(np.abs(mae_scores))))

#y_pred = cross_val_predict(model, X, y, cv=180)

#np.savetxt('leave_one_out_results.csv',[y,y_pred],delimiter=',')
#os.system('rm leave_one_out_results.csv')

#for i in range(180):
#    print ('%f,%f'%(y.to_numpy()[i],y_pred[i]),file=open('leave_one_out_results.csv','a'))

model.fit(X,y)
importances = model.feature_importances_

print (model.score(X,y))
#print (importances)
#model.fit(X,y)

included = X.columns.values
indices = np.argsort(importances)[::-1]

#from sklearn.linear_model import LinearRegression
#lr = LinearRegression()
#lr.fit(X.values,y)
#print (lr.score(X.values,y))
#crossvalidation = KFold(n_splits=179, shuffle=True, random_state=1)
#scores = cross_val_score(lr, X, y, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=1)
#mae_scores = [s for s in scores]

#print('Cross-validation results:')
#print('Folds: %i, mean MAE: %.3f' % (len(scores), np.mean(np.abs(mae_scores))))

#os.system('rm importances.csv correlation.csv')

for i in range(len(included)):
    print ('%s,%f'%(included[i],importances[i]),file=open('importances.csv','a'))
#    lr = LinearRegression()
#    print (X[included[i]].values,y.values.flatten())
#    lr.fit(X[included[i]].values.reshape(-1,1),y.values.flatten())
#    print ('%s,%f'%(included[i],lr.score(X[included[i]].values.reshape(-1,1),y.values.flatten())),file=open('correlation.csv','a'))
#    print (X[included[i]].values)

#os.system('rm mean_mendeleev_number.csv')

#for i in range(len(y.values)):
#    print ('%s,%f'%(X[['MagpieData mean MendeleevNumber']].values[i][0],y.values[i]),file=open('mean_mendeleev_number.csv','a'))

#os.system('rm dev_covalent_radius.csv')

#for i in range(len(y.values)):
#    print ('%s,%f'%(X[['MagpieData avg_dev CovalentRadius']].values[i][0],y.values[i]),file=open('dev_covalent_radius.csv','a'))

#os.system('rm nat_area.csv')

#for i in range(len(y.values)):
#    print ('%s,%f'%(X[['density']].values[i][0],y.values[i]),file=open('nat_area.csv','a'))
#print (included)
#print('training MAE = %.3f' % (mean_absolute_error(y_true=y, y_pred=model.predict(X))))

#from sklearn.metrics import r2_score

'''
input = Variable(torch.tensor(X.values)).float(); target = Variable(torch.tensor(y.values)).float()

optimizer = torch.optim.Adam(model.parameters(), lr)
loss_func = torch.nn.MSELoss()

#print (input[0])

for step in range(epochs):
    out, shared_fea = model(input)
    loss = loss_func(out, target) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#    print (loss, mae(out, target))
    print (loss,mae(out[:,0], target[:,0]), mae(out[:,1], target[:,1]))


pred = np.concatenate((out.detach().numpy(), y.values),axis=1)
np.savetxt('prediction.csv', pred, delimiter=',')

pca = decomposition.PCA(n_components = 2)
shared_fea = pca.fit_transform(shared_fea.detach().numpy())
shared_layer = np.concatenate((shared_fea, y.values),axis=1)
np.savetxt('shared_layer.csv', shared_layer, delimiter=',')
'''
