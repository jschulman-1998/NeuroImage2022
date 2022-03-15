## Import Libraries and set seaborn as theme
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import circulant
from scipy.optimize import curve_fit
from scipy.optimize import rosen, differential_evolution

A = a = 0.95
v = V = a - 0.35
c = (A+V)/2
TE=0.03
noise=float(input("Noise? ")) 
ArtVol = 0.6
Tiss = 1
X = np.array(range(120))
X2 = np.array(range(240))

Y = 2*((X/2.5)**3)*np.exp(3*(1-(X/2.5)))
Y = np.pad(Y, (10, 0), 'constant', constant_values=(0, 0))
GadIN = Y[0:len(X)]

y = []
for i in X:
    if 10<X[i]<15:
        a = 0.23
    else:
        a = 0
    y.append(a)
DeoxyIN = np.array(y)


def residue(x, f, T, t):
    return f*np.exp(-T*(x)) + (1-f)*np.exp(-t*(x))
def residue2(x, mtt):
    return np.exp(-x/mtt)
    
##Artery (Gad then dHb)
y1 = np.convolve(DeoxyIN, residue2(X,1.5))
y1 = np.delete(y1, np.s_[len(X):len(X2)])
DeoxyIN = ((metrics.auc(X, DeoxyIN))/(metrics.auc(X, y1)))*y1

##Tissue
y2 = np.convolve(GadIN, residue(X,0.92,0.68, 0.05))
y2 = np.delete(y2, np.s_[len(X):len(X2)])
GadINt = (metrics.auc(X, GadIN)/metrics.auc(X, y2))*y2

y2 = np.convolve(DeoxyIN, residue(X,0.92,0.68, 0.05))
y2 = np.delete(y2, np.s_[len(X):len(X2)])
DeoxyINt = (metrics.auc(X, DeoxyIN)/metrics.auc(X, y2))*y2

print('Theoretical CBV, MTT, CBF: ', Tiss*4, '%, ', round(metrics.auc(X, residue2(X,3)), 2), ', ', (Tiss*0.04)/round(metrics.auc(X, residue2(X,3))))

##Vein
y3 = np.convolve(GadIN, residue(X,0.863,0.68, 0.05))
y3 = np.delete(y3, np.s_[len(X):len(X2)])
GadINv = (metrics.auc(X, GadIN)/metrics.auc(X, y3))*y3

y3 = np.convolve(DeoxyIN, residue(X,0.863,0.68, 0.05))
y3 = np.delete(y3, np.s_[len(X):len(X2)])
DeoxyINv = (metrics.auc(X, DeoxyIN)/metrics.auc(X, y3))*y3

#From Zhao et al 2007 (For hematocrit of 0.44) (alternative is 20.7 from Zhao)
R2_Star_In = 13.8
##R2*0,Ex (Page 4 (Equation 6 Kamil))
R2_Star_Ex = 20.99
##R2*Hb,In (Page 6 (Equation 9b Kamil))
R2_Star_In_c_dhb = 181*(np.square(1-c+DeoxyINt)) + R2_Star_In
R2_Star_In_a_dhb = 181*(np.square(1-a+DeoxyINt)) + R2_Star_In
R2_Star_In_v_dhb = 181*(np.square(1-v+DeoxyINt)) + R2_Star_In
R2_Star_In_A_dhb = 181*(np.square(1-A+DeoxyIN)) + R2_Star_In
R2_Star_In_V_dhb = 181*(np.square(1-V+DeoxyINv)) + R2_Star_In
R2_Star_In_AT_dhb = 181*(np.square(1-A+DeoxyINt)) + R2_Star_In
R2_Star_In_VT_dhb = 181*(np.square(1-V+DeoxyINt)) + R2_Star_In
rI  = 0.246
R2_Star_In_c_gad = 181*(np.square(1-c+rI*GadINt)) + R2_Star_In
R2_Star_In_a_gad = 181*(np.square(1-a+rI*GadINt)) + R2_Star_In
R2_Star_In_v_gad = 181*(np.square(1-v+rI*GadINt)) + R2_Star_In
R2_Star_In_A_gad = 181*(np.square(1-A+rI*GadIN)) + R2_Star_In
R2_Star_In_V_gad = 181*(np.square(1-V+rI*GadINv)) + R2_Star_In
R2_Star_In_AT_gad = 181*(np.square(1-A+rI*GadINt)) + R2_Star_In
R2_Star_In_VT_gad = 181*(np.square(1-V+rI*GadINt)) + R2_Star_In

rE = 20.877
##Frequency Change based on hematocrit and oxygenation (Page 6 (Equation 10 Kamil))
vs_c_dhb = 84.796*(1-c+DeoxyINt)
vs_a_dhb = 84.796*(1-a+DeoxyINt)
vs_v_dhb = 84.796*(1-v+DeoxyINt)
vs_A_dhb = 84.796*(1-A+DeoxyIN)
vs_V_dhb = 84.796*(1-V+DeoxyINv)
vs_AT_dhb = 84.796*(1-A+DeoxyINt)
vs_VT_dhb = 84.796*(1-V+DeoxyINt)

vs_c_gad = 84.796*(1-c) + GadINt*rE
vs_a_gad = 84.796*(1-a) + GadINt*rE
vs_v_gad = 84.796*(1-v) + GadINt*rE
vs_A_gad = 84.796*(1-A) + GadIN*rE
vs_V_gad = 84.796*(1-V) + GadINv*rE
vs_AT_gad = 84.796*(1-A) + GadINt*rE
vs_VT_gad = 84.796*(1-V) + GadINt*rE

##R2*Hb,Ex (Table 2 Page 7 Kamil)
R2_Star_Ex_c_dhb = 0.03865374*vs_c_dhb
R2_Star_Ex_v_dhb = 0.04330869*vs_v_dhb
R2_Star_Ex_a_dhb = 0.04330869*vs_a_dhb
R2_Star_Ex_V_dhb = 0.07976623*vs_V_dhb
R2_Star_Ex_A_dhb = 0.04330869*vs_A_dhb
R2_Star_Ex_A_90_t_dhb = 0.07976623*vs_AT_dhb
R2_Star_Ex_V_t_dhb = 0.04330869*vs_VT_dhb
R2_Star_Micro_Ex_dhb = Tiss*1*R2_Star_Ex_c_dhb + Tiss*1*R2_Star_Ex_v_dhb + Tiss*1*R2_Star_Ex_V_t_dhb + Tiss*0.5*R2_Star_Ex_a_dhb + Tiss*0.5*R2_Star_Ex_A_90_t_dhb

R2_Star_Ex_c_gad = 0.03865374*vs_c_gad
R2_Star_Ex_v_gad = 0.04330869*vs_v_gad
R2_Star_Ex_a_gad = 0.04330869*vs_a_gad
R2_Star_Ex_V_gad = 0.07976623*vs_V_gad
R2_Star_Ex_A_gad = 0.04330869*vs_A_gad
R2_Star_Ex_A_90_t_gad = 0.07976623*vs_AT_gad
R2_Star_Ex_V_t_gad = 0.04330869*vs_VT_gad
R2_Star_Micro_Ex_gad = Tiss*1*R2_Star_Ex_c_gad + Tiss*1*R2_Star_Ex_v_gad + Tiss*1*R2_Star_Ex_V_t_gad + Tiss*0.5*R2_Star_Ex_a_gad + Tiss*0.5*R2_Star_Ex_A_90_t_gad

##Deoxy Signal
s_Micro_dhb = (1-Tiss*0.04)*np.exp(-(R2_Star_Ex+R2_Star_Micro_Ex_dhb)*TE) + Tiss*0.01*np.exp(-(R2_Star_In_c_dhb)*TE) + Tiss*0.01*np.exp(-(R2_Star_In_v_dhb)*TE) + Tiss*0.005*np.exp(-(R2_Star_In_a_dhb)*TE) + Tiss*0.005*np.exp(-(R2_Star_In_AT_dhb)*TE) + Tiss*0.01*np.exp(-(R2_Star_In_VT_dhb)*TE)
l = []
for i in s_Micro_dhb:
    b = i + np.random.normal(0, noise*s_Micro_dhb[0])
    l.append(b)
S_Micro_dhb = np.array(l)

s_Artery_dhb = (1-ArtVol)*np.exp(-(R2_Star_Ex+R2_Star_Ex_A_dhb*(ArtVol*100))*TE) + ArtVol*np.exp(-(R2_Star_In_A_dhb)*TE)
o = []
for i in s_Artery_dhb:
    b = i + np.random.normal(0, noise*s_Artery_dhb[0])
    o.append(b)
S_Artery_dhb = np.array(o)

s_Vein_dhb = (1-ArtVol)*np.exp(-(R2_Star_Ex+R2_Star_Ex_V_dhb*(ArtVol*100))*TE) + ArtVol*np.exp(-(R2_Star_In_V_dhb)*TE)
O = []
for i in s_Vein_dhb:
    b = i + np.random.normal(0, noise*s_Vein_dhb[0])
    O.append(b)
S_Vein_dhb = np.array(O)

##Gadolinium Signal 
s_Micro_gad = (1-Tiss*0.04)*np.exp(-(R2_Star_Ex+R2_Star_Micro_Ex_gad)*TE) + Tiss*0.01*np.exp(-(R2_Star_In_c_gad)*TE) + Tiss*0.01*np.exp(-(R2_Star_In_v_gad)*TE) + Tiss*0.005*np.exp(-(R2_Star_In_a_gad)*TE) + Tiss*0.005*np.exp(-(R2_Star_In_AT_gad)*TE) + Tiss*0.01*np.exp(-(R2_Star_In_VT_gad)*TE)
l = []
for i in s_Micro_gad:
    b = i + np.random.normal(0, noise*s_Micro_gad[0])
    l.append(b)
S_Micro_gad = np.array(l)

s_Artery_gad = (1-ArtVol)*np.exp(-(R2_Star_Ex+R2_Star_Ex_A_gad*(ArtVol*100))*TE) + ArtVol*np.exp(-(R2_Star_In_A_gad)*TE)
o = []
for i in s_Artery_gad:
    b = i + np.random.normal(0, noise*s_Artery_gad[0])
    o.append(b)
S_Artery_gad = np.array(o)

s_Vein_gad = (1-ArtVol)*np.exp(-(R2_Star_Ex+R2_Star_Ex_V_gad*(ArtVol*100))*TE) + ArtVol*np.exp(-(R2_Star_In_V_gad)*TE)
O = []
for i in s_Vein_gad:
    b = i + np.random.normal(0, noise*s_Vein_gad[0])
    O.append(b)
S_Vein_gad = np.array(O)

#plt.plot(range(-10,50),S_Artery_gad[0:60],color='red', label='Gadolinium Arterial')
#plt.plot(range(-10,50),S_Micro_gad[0:60], label='Gadolinium Tissue')
#plt.plot(range(-10,50),S_Vein_gad[0:60], label='Gadolinium Venous')

#plt.plot(range(-10,50),S_Artery_dhb[0:60], label='DeoxyHemoglobin Arterial')
#plt.plot(range(-10,50),S_Micro_dhb[0:60], label='DeoxyHemoglobin Tissue')
#plt.plot(range(-10,50),S_Vein_dhb[0:60], label='DeoxyHemoglobin Venous')

##Delta R2* Curves
GadR2Art = (-1/TE)*np.log((S_Artery_gad/S_Artery_gad[0]))
GadR2T = (-1/TE)*np.log((S_Micro_gad/S_Micro_gad[0]))
GadR2Vein = (-1/TE)*np.log((S_Vein_gad/S_Vein_gad[0]))

OxR2Art = (-1/TE)*np.log((S_Artery_dhb/S_Artery_dhb[0]))
OxR2T = (-1/TE)*np.log((S_Micro_dhb/S_Micro_dhb[0]))
OxR2Vein = (-1/TE)*np.log((S_Vein_dhb/S_Vein_dhb[0]))


plt.plot(range(-10, 50),GadR2Art[0:60], color='red', label='Gadolinium Arterial')
#plt.plot(range(-10, 50),GadR2T[0:60], label='Gadolinium Tissue')
#plt.plot(range(-10, 50),GadR2Vein[0:60], label='Gadolinium Venous')

plt.plot(range(-10,50),OxR2Art[0:60], color='blue', label='DeoxyHemoglobin Arterial')
#plt.plot(range(-10,50),OxR2T[0:60], label='DeoxyHemoglobin Tissue')
#plt.plot(range(-10,50),OxR2Vein[0:60], label='DeoxyHemoglobin Venous')

##CBV Calculations
if noise == 0:
    Area_Tissue = metrics.auc(X, GadR2T)
    Area_Arterial = metrics.auc(X, GadR2Art)
else:
    Area_Tissue = metrics.auc(X[0,20], GadR2T[0,20])
    Area_Arterial = metrics.auc(X[0,20], GadR2Art[0,20])
CBVgad=Area_Tissue/Area_Arterial
print("CBVgad = ", round(CBVgad, 4))

if noise == 0:
    Area_Tissue = metrics.auc(X, OxR2T)
    Area_Arterial = metrics.auc(X, OxR2Art)
else:
    Area_Tissue = metrics.auc(X[10,32], OxR2T[10,32])
    Area_Arterial = metrics.auc(X[10,32], OxR2Art[10,32])
CBVox=Area_Tissue/Area_Arterial
print("CBVdhb = ", round(CBVox, 4))


##CBF and MTT Calculations
AIF1 = []
for i in range(len(X)):
    a = GadR2Art
    A = np.delete(a, np.s_[len(X)-i:len(X)])
    b = np.pad(A, (i, 0), 'constant', constant_values=(0, 0))
    B = np.array(b)
    AIF1.append(B)
AIFgad = np.transpose(AIF1)

AIF2 = []
for i in range(len(X)):
    a = OxR2Art
    A = np.delete(a, np.s_[len(X)-i:len(X)])
    b = np.pad(A, (i, 0), 'constant', constant_values=(0, 0))
    B = np.array(b)
    AIF2.append(B)
AIFox = np.transpose(AIF2)

##Determine No Noise Residue Properties
if noise == 0:
    b1, b2, b3 = randomized_svd(AIFgad, n_components = 120)
    B1 = b1.T
    B2=np.diag(b2)
    B2i = np.linalg.inv(B2)
    B3 = b3.T
    Residue1 = np.transpose([np.matmul(B3, np.matmul(B2i, np.matmul(B1, np.transpose(GadR2T))))])
    max_value1 = max([max(i) for i in Residue1[0:20]])
    print("CBFgad: ", round(max_value1,4))
    print("MTTgad: ", round(CBVgad/max_value1,4))

    b1, b2, b3 = randomized_svd(AIFox, n_components = 120)
    B1 = b1.T
    B2=np.diag(b2)
    B2i = np.linalg.inv(B2)
    B3 = b3.T
    Residue1 = np.transpose([np.matmul(B3, np.matmul(B2i, np.matmul(B1, np.transpose(OxR2T))))])
    max_value1 = max([max(i) for i in Residue1[0:20]])
    print("CBFox: ", round(max_value1,4))
    print("MTTox: ", round(CBVox/max_value1,4))
else:
    omega = 2.858
    c1, c2, c3 = randomized_svd(AIFgad, n_components = len(X))
    truncation = np.max(np.where(c2 > omega*np.median(c2)))
    d1, d2, d3 = randomized_svd(AIFgad, n_components = truncation)
    D1 = d1.T
    D2=np.diag(d2)
    D2i = np.linalg.inv(D2)
    D3 = d3.T
    Residue2 = np.transpose([np.matmul(D3, np.matmul(D2i, np.matmul(D1, np.transpose(GadR2T))))])
    max_value2 = CBVgad*(max([max(i) for i in Residue2])/metrics.auc(X, Residue2))
    print("CBFgad: ", round(max_value2,4))
    print("MTTgad: ", round(CBVgad/max_value2,4))

    c1, c2, c3 = randomized_svd(AIFox, n_components = len(X))
    truncation = np.max(np.where(c2 > omega*np.median(c2)))
    d1, d2, d3 = randomized_svd(AIFox, n_components = truncation)
    D1 = d1.T
    D2=np.diag(d2)
    D2i = np.linalg.inv(D2)
    D3 = d3.T
    Residue2 = np.transpose([np.matmul(D3, np.matmul(D2i, np.matmul(D1, np.transpose(OxR2T))))])
    max_value2 = CBVox*(max([max(i) for i in Residue2])/metrics.auc(X, Residue2))
    print("CBFox: ", round(max_value2,4))
    print("MTTox: ", round(CBVox/max_value2,4))


plt.xlabel('Time after Induced Contrast (s)', fontsize=12)
plt.ylabel('Relaxation Rate (1/s)', fontsize=12)
#plt.title('Contrast Relaxation', fontsize=16)
plt.legend(title="Legend")
plt.show()