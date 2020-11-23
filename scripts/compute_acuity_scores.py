'''
This script computes the acuity scores corresponding to the Sepsis patient cohort extracted with 
the procedure provided at: https://github.com/microsoft/mimic_sepsis using the raw features.

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

'''
import sys
import os
import time

import pandas as pd
import numpy as np

save_dir = 'data/sepsis_mimiciii/'
acuity_file = 'acuity_scores.csv'

data_file = os.path.join(save_dir, 'sepsis_final_data_RAW_withTimes.csv')

table = pd.read_csv(data_file)

obs_cols = ['o:GCS', 'o:HR', 'o:SysBP',
       'o:MeanBP', 'o:DiaBP', 'o:RR', 'o:Temp_C', 'o:FiO2_1', 'o:Potassium',
       'o:Sodium', 'o:Chloride', 'o:Glucose', 'o:Magnesium', 'o:Calcium',
       'o:Hb', 'o:WBC_count', 'o:Platelets_count', 'o:PTT', 'o:PT',
       'o:Arterial_pH', 'o:paO2', 'o:paCO2', 'o:Arterial_BE', 'o:HCO3',
       'o:Arterial_lactate', 'o:SIRS', 'o:Shock_Index',
       'o:PaO2_FiO2', 'o:cumulated_balance', 'o:SpO2', 'o:BUN', 'o:Creatinine',
       'o:SGOT', 'o:SGPT', 'o:Total_bili', 'o:INR', 'o:input_total',
       'o:input_4hourly', 'o:output_total', 'o:output_4hourly']

############################################################################
#         FUNCTIONS TO CALCULATE THE ACUITY SCORES NOT EXTRACTED
############################################################################

def calc_sapsii(df):
    """ Calculate the SAPSII score provided the dataframe of raw patient features. """
    age_values = np.array([0, 7, 12, 15, 16, 18])
    hr_values = np.array([11, 2, 0, 4, 7])
    bp_values = np.array([13, 5, 0, 2])
    temp_values = np.array([0, 3])
    o2_values = np.array([11, 9, 6])
    output_values = np.array([11, 4, 0])
    bun_values = np.array([0, 6, 10])
    wbc_values = np.array([12, 0, 3])
    k_values = np.array([3, 0, 3])
    na_values = np.array([5, 0, 1])
    hco3_values = np.array([5, 3, 0])
    bili_values = np.array([0, 4, 9])
    gcs_values = np.array([26, 13, 7, 5, 0])
    
    sapsii = np.zeros((df.shape[0],1))
    
    cols = ['o:age','o:HR','o:SysBP','o:Temp_C','o:PaO2_FiO2','o:output_4hourly','o:BUN','o:WBC_count','o:Potassium','o:Sodium','o:HCO3','o:Total_bili','o:GCS']
    tt = df[cols]
    tt.loc[:,'o:age'] = tt['o:age'].values/365.24
    
    age = np.array([ tt.iloc[:,0]<40, (tt.iloc[:,0]>=40)&(tt.iloc[:,0]<60), (tt.iloc[:,0]>=60)&(tt.iloc[:,0]<70), (tt.iloc[:,0]>=70)&(tt.iloc[:,0]<75), (tt.iloc[:,0]>=75)&(tt.iloc[:,0]<80), tt.iloc[:,0]>=80 ])
    hr = np.array([ tt.iloc[:,1]<40, (tt.iloc[:,1]>=40)&(tt.iloc[:,1]<70), (tt.iloc[:,1]>=70)&(tt.iloc[:,1]<120), (tt.iloc[:,1]>=120)&(tt.iloc[:,1]<160), tt.iloc[:,1]>=160 ])
    bp = np.array([ tt.iloc[:,2]<70, (tt.iloc[:,2]>=70)&(tt.iloc[:,2]<100), (tt.iloc[:,2]>=100)&(tt.iloc[:,2]<200), tt.iloc[:,2]>=200 ])
    temp = np.array([ tt.iloc[:,3]<39, tt.iloc[:,3]>=39 ])
    o2 = np.array([ tt.iloc[:,4]<100, (tt.iloc[:,4]>=100)&(tt.iloc[:,4]<200), tt.iloc[:,4]>=200 ])
    out = np.array([ tt.iloc[:,5]<500, (tt.iloc[:,5]>=500)&(tt.iloc[:,5]<1000), tt.iloc[:,5]>=1000 ])
    bun = np.array([ tt.iloc[:,6]<28, (tt.iloc[:,6]>=28)&(tt.iloc[:,6]<84), tt.iloc[:,6]>=84 ])
    wbc = np.array([ tt.iloc[:,7]<1, (tt.iloc[:,7]>=1)&(tt.iloc[:,7]<20), tt.iloc[:,7]>=20 ])
    k = np.array([ tt.iloc[:,8]<3, (tt.iloc[:,8]>=3)&(tt.iloc[:,8]<5), tt.iloc[:,8]>=5 ])
    na = np.array([ tt.iloc[:,9]<125, (tt.iloc[:,9]>=125)&(tt.iloc[:,9]<145), tt.iloc[:,9]>=145 ])
    hco3 = np.array([ tt.iloc[:,10]<15, (tt.iloc[:,10]>=15)&(tt.iloc[:,10]<20), tt.iloc[:,10]>=20 ])
    bili = np.array([ tt.iloc[:,11]<4, (tt.iloc[:,11]>=4)&(tt.iloc[:,11]<6), tt.iloc[:,11]>=6 ])
    gcs = np.array([ tt.iloc[:,12]<6, (tt.iloc[:,12]>=6)&(tt.iloc[:,12]<9), (tt.iloc[:,12]>=9)&(tt.iloc[:,12]<11), (tt.iloc[:,12]>=11)&(tt.iloc[:,12]<14), tt.iloc[:,12]>=14 ])
    
    for ii in range(df.shape[0]):
        sapsii[ii] = max(age_values[age[:,ii]], default=0) + max(hr_values[hr[:,ii]], default=0) + max(bp_values[bp[:,ii]], default=0) + max(temp_values[temp[:,ii]], default=0) + max(o2_values[o2[:,ii]]*df.loc[ii,'o:mechvent'], default=0) + max(output_values[out[:,ii]], default=0) + max(bun_values[bun[:,ii]], default=0) + max(wbc_values[wbc[:,ii]], default=0) + max(k_values[k[:,ii]], default=0) + max(na_values[na[:,ii]], default=0) + max(hco3_values[hco3[:,ii]], default=0) + max(bili_values[bili[:,ii]], default=0) + max(gcs_values[gcs[:,ii]], default=0)
    return sapsii

def calc_oasis(df):
    """ Calculate the OASIS score provided the dataframe of raw patient features. """
    age_values = np.array([0, 3, 6, 9, 7])
    bp_values = np.array([4, 3, 2, 0, 3])
    gcs_values = np.array([10, 4, 3, 0])
    hr_values = np.array([4, 0, 1, 3, 6])
    rr_values = np.array([10, 1, 0, 1, 6, 9])
    temp_values = np.array([3, 4, 2, 2, 6])
    output_values = np.array([10, 5, 1, 0, 8])
    vent_value = 9
    
    oasis = np.zeros((df.shape[0],1))
    
    cols = ['o:age','o:MeanBP','o:GCS','o:HR','o:RR','o:Temp_C','o:output_4hourly']
    tt = df[cols]
    tt.loc[:,'o:age'] = tt['o:age'].values/365.24 # Convert the age to years
    
    age = np.array([ tt.iloc[:,0]<24, (tt.iloc[:,0]>=24)&(tt.iloc[:,0]<=53), (tt.iloc[:,0]>53)&(tt.iloc[:,0]<=77), (tt.iloc[:,0]>77)&(tt.iloc[:,0]<=89), tt.iloc[:,0]>89 ])
    bp = np.array([ tt.iloc[:,1]<20.65, (tt.iloc[:,1]>=20.65)&(tt.iloc[:,1]<51), (tt.iloc[:,1]>=51)&(tt.iloc[:,1]<61.33), (tt.iloc[:,1]>=61.33)&(tt.iloc[:,1]<143.44), tt.iloc[:,1]>=143.44 ])
    gcs = np.array([ tt.iloc[:,2]<=7, (tt.iloc[:,2]>7)&(tt.iloc[:,1]<14), tt.iloc[:,1]==14, tt.iloc[:,1]>14 ])
    hr = np.array([ tt.iloc[:,3]<33, (tt.iloc[:,3]>=33)&(tt.iloc[:,3]<89), (tt.iloc[:,3]>=89)&(tt.iloc[:,3]<106), (tt.iloc[:,3]>=106)&(tt.iloc[:,3]<=125), tt.iloc[:,3]>125 ])
    rr = np.array([ tt.iloc[:,4]<6, (tt.iloc[:,4]>=6)&(tt.iloc[:,4]<13), (tt.iloc[:,4]>=13)&(tt.iloc[:,4]<22), (tt.iloc[:,4]>=22)&(tt.iloc[:,4]<30), (tt.iloc[:,4]>=30)&(tt.iloc[:,4]<44), tt.iloc[:,4]>=44 ])
    temp = np.array([ tt.iloc[:,5]<33.22, (tt.iloc[:,5]>=33.22)&(tt.iloc[:,5]<35.93), (tt.iloc[:,5]>=35.93)&(tt.iloc[:,5]<36.89), (tt.iloc[:,5]>=36.89)&(tt.iloc[:,5]<=39.88), tt.iloc[:,5]>39.88 ])
    out = np.array([ tt.iloc[:,6]<671.09, (tt.iloc[:,6]>=671.09)&(tt.iloc[:,6]<1427), (tt.iloc[:,6]>=1427)&(tt.iloc[:,6]<=2514), (tt.iloc[:,6]>2514)&(tt.iloc[:,6]<=6896), tt.iloc[:,6]>6896 ])
    vent = (vent_value*df['o:mechvent']).values
    
    for ii in range(df.shape[0]):
        oasis[ii] = max(age_values[age[:,ii]], default=0) + max(bp_values[bp[:,ii]], default=0) + max(gcs_values[gcs[:,ii]], default=0) + max(hr_values[hr[:,ii]], default=0) + max(rr_values[rr[:,ii]], default=0) + max(temp_values[temp[:,ii]], default=0) + max(output_values[out[:,ii]], default=0) + vent[ii]
        
    return oasis

############################################################################
#                  ORGANIZE TABLE AND COMPUTE ACUITY SCORES
############################################################################

# Compute OASIS
table['c:oasis'] = calc_oasis(table)

# Compute SAPSII
table['c:sapsii'] = calc_sapsii(table)


# Isolate only the columns we want to keep from `table`
keeping = table[['traj', 'step', 'o:SOFA', 'c:oasis', 'c:sapsii']]
keeping = keeping.rename(columns={"o:SOFA": "c:SOFA", "c:oasis": "c:OASIS", "c:sapsii":"c:SAPSii"})

# Save off the acuity scores for use later 
keeping.to_csv(os.path.join(save_dir,acuity_file))