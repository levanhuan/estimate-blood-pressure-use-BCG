import glob
import os
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

folder_path = 'D:\\courses\\AI nang cao\\numpy2'
file_pattern = os.path.join(folder_path, '*.npy')
file_names=[]
pcg2dArr=[]
bcg2dArr=[]
bpArr=[]
i=0

#load data: creat an Array have same length of BCG, PCG 30s signal and blood pressure
for file_path in glob.glob(file_pattern):
    if os.path.isfile(file_path):
        if (i%2==0):
            pcg2dArr.append(np.load(file_path))
        else:
            bcg2dArr.append(np.load(file_path))
            words=file_path.split('_')
            bpArr.append(float(words[2]))
        i+=1
        
#filter
fs = 1000  
low_cutoff = 34 
high_cutoff = 50  
filter_order = 501 
nyquist = 0.5 * fs
low_cutoff_normalized = low_cutoff / nyquist
high_cutoff_normalized = high_cutoff / nyquist
coefficients = firwin(filter_order, [low_cutoff_normalized, high_cutoff_normalized], pass_zero=False, window='blackman')

low_cutoff = 1
high_cutoff = 10 
filter_order = 501 
low_cutoff_normalized = low_cutoff / nyquist
high_cutoff_normalized = high_cutoff / nyquist
coefficients2 = firwin(filter_order, [low_cutoff_normalized, high_cutoff_normalized], pass_zero=False, window='blackman')

pFilter=[]
bFilter=[]
for i in range(len(pcg2dArr)):
    pFilter.append(lfilter(coefficients, 1.0,  pcg2dArr[i]))
    bFilter.append(lfilter(coefficients2, 1.0,  bcg2dArr[i]))
#normalize signal-put signal to 0-1 range
pNormal=[]
bNormal=[]

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

for i in range(len(pFilter)):
    original_data=np.array(pFilter[i])
    normalized_data = min_max_scaling(original_data)
    pNormal.append(normalized_data)
    
    original_data=np.array(bFilter[i])
    normalized_data = min_max_scaling(original_data)
    bNormal.append(normalized_data)

#find s1,s2,i,j,k in each cardiac cycle and build observation stimulately
X=[]
y=[]
numOfmeasure=len(pNormal)
#duyet cac phep do
for i in range(numOfmeasure):
    bp=bpArr[i]
    pSignal=pNormal[i]
    bSignal=bNormal[i]
    numOfsample=len(pSignal)//800
    #duyet cac doan 800 mau
    for j in range(numOfsample):
        pSample=pSignal[j*800:(j+1)*800]
        bSample=bSignal[j*800:(j+1)*800]
        found=True
        #find I,J,K index
        J_index=np.argmax(bSample)
        gradient_sign_changes = np.diff(np.sign(np.gradient(bSample))) 
        local_minima_indices = np.where(gradient_sign_changes > 0)[0] 
        for k in range(len(local_minima_indices)):
            if local_minima_indices[k]-J_index>0:
                break
        I_index=local_minima_indices[k-1]
        K_index=local_minima_indices[k]
        if (I_index>=J_index or K_index<=J_index):
            found=False
        #find S1,S2 index
        gradient_sign_changes = np.diff(np.sign(np.gradient(pSample))) 
        local_maxima_indices = np.where(gradient_sign_changes < 0)[0]
        for l in range(len(local_maxima_indices)):
            if local_maxima_indices[l]-I_index>0:
                break
        S1_index=local_maxima_indices[l-1]
        for m in range(len(local_maxima_indices)):
            if local_maxima_indices[m]-K_index>0:
                break
        S2_index=local_maxima_indices[m]
        if (S1_index>=I_index or S2_index<=K_index):
            found=False
        xi0=(J_index-S1_index)*0.001
        xi1=(S2_index-J_index)*0.001
        xi2=(I_index-S1_index)*0.001
        xi3=(S2_index-I_index)*0.001
        xi4=(K_index-S1_index)*0.001
        xi5=(S2_index-K_index)*0.001
        xi6=(J_index-I_index)*0.001
        xi7=(K_index-J_index)*0.001
        xi8=(bSample[J_index]-bSample[I_index])/xi6
        xi9=(bSample[J_index]-bSample[K_index])/xi7
        xi=[xi0,xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9]
        if found:
            X.append(xi)
            y.append(bp)       
#eliminate outlier
column0=[]
column4=[]
column6=[]
for i in range (len(X)):
    column0.append(X[i][0])
    column4.append(X[i][4])
    column6.append(X[i][6])
meanS1_J=sum(column0)/len(column0)
stdS1_J=np.std(column0)

meanS1_K=sum(column4)/len(column4)
stdS1_K=np.std(column4)

meanI_J=sum(column6)/len(column6)
stdI_J=np.std(column6)

XD2=[]
yD2=[]
for i in range (len(X)):
    if (abs(X[i][0]-meanS1_J)<=2*stdS1_J) and (abs(X[i][4]-meanS1_K)<=2*stdS1_K) and (abs(X[i][6]-meanI_J)<=2*stdI_J):
        XD2.append(X[i])
        yD2.append(y[i])
#print(len(XD2))
#training
X_train, X_test, y_train, y_test = train_test_split(XD2, yD2, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=210,random_state=0)
regressor.fit(X_train, y_train)
yPredict=regressor.predict(X_test)
error=mae(y_test,yPredict)
print('Mean absolute error is:')
print(error)

