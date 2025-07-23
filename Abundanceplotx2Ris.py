import numpy as np
import matplotlib.pyplot as plt

# === Load finab.dat ===
# Assumes the file starts with a header or line numbers (we'll skip them)
data = np.loadtxt("finab.dat", comments="#", usecols=(0, 4))  # Columns: A, Xi=4, 3 = Yi
data2 = np.loadtxt("../runNi57Cu58_0/finab.dat", comments="#", usecols=(0, 4))
dataSD = np.loadtxt("../runNi57Cu58_Lx10/finab.dat", comments="#", usecols=(0, 4))
dataSU = np.loadtxt("../runNi57Cu58_Hx10/finab.dat", comments="#", usecols=(0, 4))
dataOf = np.loadtxt("../runNi57Cu58_00/finab.dat", comments="#", usecols=(0, 4))
dataCCSN = np.loadtxt("../runNi57Cu58_CCSN/finab.dat", comments="#", usecols=(0, 4))
dataCCSNnu = np.loadtxt("../runNi57Cu58_CCSN_neu/finab.dat", comments="#", usecols=(0, 4))
datanuoff = np.loadtxt("../runNi57Cu58_MRSN_neuOff/finab.dat", comments="#", usecols=(0, 4))
dataCCSNneup = np.loadtxt("../runCCSN_neup/finab.dat", comments="#", usecols=(0, 4))
dataMRSN = np.loadtxt("../runMRSN_short_51325/finab.dat", comments="#", usecols=(0, 4))
datanoneuMRSN = np.loadtxt("../runMRSN_short_noneu_51325/finab.dat", comments="#", usecols=(0, 4))
dataMRSN_NipgOff = np.loadtxt("../runMRSN_Ni57pgOff/finab.dat", comments="#", usecols=(0, 4))
dataCCSN_NipgOff = np.loadtxt("../runtestCCSN_noNi57pg/finab.dat", comments="#", usecols=(0, 4))
dataCCSN_Nipg = np.loadtxt("../runtestCCSN_Ni57pg/finab.dat", comments="#", usecols=(0, 4))
dataCCSN_Nipg2 = np.loadtxt("../runtestCCSN_Ni57pg_2/finab.dat", comments="#", usecols=(0, 4))
dataCCSN_Nipgx100 = np.loadtxt("../runtestCCSN_Ni57pg_x100/finab.dat", comments="#", usecols=(0, 4))
dataCCSN_Nipgd100 = np.loadtxt("../runtestCCSN_Ni57pg_div100/finab.dat", comments="#", usecols=(0, 4))



#\\wsl.localhost\Ubuntu\usr\Sean\Programs\WinNet\runs
# Split into arrays
A = data[:, 0].astype(int)
Aj= data2[:, 0].astype(int)
Al= dataSD[:, 0].astype(int)
Ah= dataSU[:, 0].astype(int)
Aof= dataOf[:, 0].astype(int)
Accsn= dataCCSN[:, 0].astype(int)
Accsnnu= dataCCSNnu[:, 0].astype(int)
Aneuoff= datanuoff[:, 0].astype(int)
Accsnnup= dataCCSNneup[:, 0].astype(int)
AMRSN = dataMRSN[:, 0].astype(int)
AnoMRSN = datanoneuMRSN[:, 0].astype(int)
AnoMRSN_niOff = dataMRSN_NipgOff[:, 0].astype(int)
AnoCCSN_niOff = dataCCSN_NipgOff[:, 0].astype(int)
AnoCCSN_ni = dataCCSN_Nipg[:, 0].astype(int)
AnoCCSN_ni2 = dataCCSN_Nipg2[:, 0].astype(int)
AnoCCSN_nix100 = dataCCSN_Nipgx100[:, 0].astype(int)
AnoCCSN_nid100 = dataCCSN_Nipgd100[:, 0].astype(int)

#print(AnoMRSN)

Xi = data[:, 1]
Xj = data2[:, 1]
Xl = dataSD[:, 1]
Xh = dataSU[:, 1]
Xof = dataOf[:, 1]
Xccsn = dataCCSN[:, 1]
Xccsnnu = dataCCSNnu[:, 1]
Xneuoff = datanuoff[:, 1]
Xccsnnup = dataCCSNneup[:, 1]
XMRSN = dataMRSN[:, 1]
XnoMRSN = datanoneuMRSN[:, 1]
XnoMRSN_niOff =dataMRSN_NipgOff[:, 1]
XnoCCSN_niOff =dataCCSN_NipgOff[:, 1]
XnoCCSN_ni =dataCCSN_Nipg[:, 1]
XnoCCSN_ni2 =dataCCSN_Nipg2[:, 1]
XnoCCSN_nix100 =dataCCSN_Nipgx100[:, 1]
XnoCCSN_nid100 =dataCCSN_Nipgd100[:, 1]

#print(XnoMRSN)
# Sort by mass number for nice plotting
sorted_indices = np.argsort(A)
A = A[sorted_indices]
Xi = Xi[sorted_indices]


sort_j = np.argsort(Aj)
Aj = Aj[sort_j]
Xj = Xj[sort_j]


sort_l = np.argsort(Al)
Al = Al[sort_l]
Xl = Xl[sort_l]


sort_h = np.argsort(Ah)
Ah = Ah[sort_h]
Xh = Xh[sort_h]


sort_of = np.argsort(Aof)
Aof = Aof[sort_of]
Xof = Xof[sort_of]

sort_ccsn = np.argsort(Accsn)
Accsn = Accsn[sort_ccsn]
Xccsn = Xccsn[sort_ccsn]


sort_ccsnnu = np.argsort(Accsnnu)
Accsnnu = Accsnnu[sort_ccsnnu]
Xccsnnu = Xccsnnu[sort_ccsnnu]


sort_neuoff = np.argsort(Aneuoff)
Aneuoff = Aneuoff[sort_neuoff]
Xneuoff = Xneuoff[sort_neuoff]


sort_ccsnnup = np.argsort(Accsnnup)
Accsnnup = Accsnnup[sort_ccsnnup]
Xccsnnup = Xccsnnup[sort_ccsnnup]

sort_MRSNshort = np.argsort(AMRSN)
AMRSN = AMRSN[sort_MRSNshort]
XMRSN = XMRSN[sort_MRSNshort]

sort_MRSNshortno = np.argsort(AnoMRSN)
AnoMRSN = AnoMRSN[sort_MRSNshortno]
XnoMRSN = XnoMRSN[sort_MRSNshortno]


sort_MRSNshortnoNi57 = np.argsort(AnoMRSN_niOff)
AnoMRSN_niOff = AnoMRSN_niOff[sort_MRSNshortnoNi57]
XnoMRSN_niOff = XnoMRSN_niOff[sort_MRSNshortnoNi57]


sort_CCSNshortnoNi57 = np.argsort(AnoCCSN_niOff)
AnoCCSN_niOff = AnoCCSN_niOff[sort_CCSNshortnoNi57]
XnoCCSN_niOff = XnoCCSN_niOff[sort_CCSNshortnoNi57]


sort_CCSNshortNi57 = np.argsort(AnoCCSN_ni)
AnoCCSN_ni = AnoCCSN_ni[sort_CCSNshortNi57]
XnoCCSN_ni = XnoCCSN_ni[sort_CCSNshortNi57]


sort_CCSNshortNi572 = np.argsort(AnoCCSN_ni2)
AnoCCSN_ni2 = AnoCCSN_ni2[sort_CCSNshortNi572]
XnoCCSN_ni2 = XnoCCSN_ni2[sort_CCSNshortNi572]


sort_CCSNshortNi57x100 = np.argsort(AnoCCSN_nix100)
AnoCCSN_nix100 = AnoCCSN_nix100[sort_CCSNshortNi57x100]
XnoCCSN_nix100 = XnoCCSN_nix100[sort_CCSNshortNi57x100]


sort_CCSNshortNi57d100 = np.argsort(AnoCCSN_nid100)
AnoCCSN_nid100 = AnoCCSN_nid100[sort_CCSNshortNi57d100]
XnoCCSN_nid100 = XnoCCSN_nid100[sort_CCSNshortNi57d100]




def sum_by_A(data):
    A_raw = data[:, 0].astype(int)
    X_raw = data[:, 1]
    unique_A = sorted(set(A_raw))
    A_sum = []
    X_sum = []
    for a in unique_A:
        mask = (A_raw == a)
        A_sum.append(a)
        X_sum.append(np.sum(X_raw[mask]))
    return np.array(A_sum), np.array(X_sum)


A, Xi = sum_by_A(data)
Aj, Xj = sum_by_A(data2)
Al, Xl = sum_by_A(dataSD)
Ah, Xh = sum_by_A(dataSU)
Aof, Xof = sum_by_A(dataOf)
Accsn, Xccsn = sum_by_A(dataCCSN)
Accsnnu, Xccsnnu = sum_by_A(dataCCSNnu)
Aneuoff, Xneuoff = sum_by_A(datanuoff)
Accsnnup, Xccsnnup = sum_by_A(dataCCSNneup)
AMRSN, XMRSN = sum_by_A(dataMRSN)
AnoMRSN, XnoMRSN = sum_by_A(datanoneuMRSN)
AnoMRSN_niOff, XnoMRSN_niOff = sum_by_A(dataMRSN_NipgOff)
AnoCCSN_niOff, XnoCCSN_niOff = sum_by_A(dataCCSN_NipgOff)
AnoCCSN_ni, XnoCCSN_ni = sum_by_A(dataCCSN_Nipg)
AnoCCSN_ni2, XnoCCSN_ni2 = sum_by_A(dataCCSN_Nipg2)
AnoCCSN_nix100, XnoCCSN_nix100 = sum_by_A(dataCCSN_Nipgx100)
AnoCCSN_nid100, XnoCCSN_nid100 = sum_by_A(dataCCSN_Nipgd100)


scale = 1e0

#MRSN_dict = dict(zip(AMRSN, XMRSN * scale))
#noMRSN_niOff_dict = dict(zip(AnoMRSN_niOff, XnoMRSN_niOff * scale))

#common_A = sorted(set(MRSN_dict) & set(noMRSN_niOff_dict))
#common_A = []
#delta_X = []
#delta_X = [noMRSN_niOff_dict[a] / MRSN_dict[a] for a in common_A]
#for a, x in zip(AMRSN, XMRSN):
#	if a in noMRSN_dict:
#		common_A.append(a)
#		delta_X.append((noMRSN_dict[a]*1e18)-(x*1e18))


CCSN_dictx100 = dict(zip(AnoCCSN_nix100, XnoCCSN_nix100 * scale))
CCSN_dictd100 = dict(zip(AnoCCSN_nid100, XnoCCSN_nid100 * scale))
#CCSN_dict = dict(zip(AnoCCSN_ni, XnoCCSN_ni * scale))
#CCSN_dict2 = dict(zip(AnoCCSN_ni2, XnoCCSN_ni2 * scale))
#noCCSN_niOff_dict = dict(zip(AnoCCSN_niOff, XnoCCSN_niOff * scale))
#common_A = sorted(set(CCSN_dict) & set(noCCSN_niOff_dict))
#delta_X = [noCCSN_niOff_dict[a] / CCSN_dict[a] for a in common_A]
common_A = sorted(set(CCSN_dictx100) & set(CCSN_dictd100))
delta_X = [ CCSN_dictx100[a] / CCSN_dictd100[a] for a in common_A]
delta_X1 = [ CCSN_dictx100[a]  for a in common_A]
delta_X2 = [  CCSN_dictd100[a] for a in common_A]

# === Plotting ===
plt.figure(figsize=(10, 6))
#plt.semilogy(Accsn, Xccsn, 'p-',color = 'blue', markersize=6, label='Mass Fraction $CCSN_ps$')
#plt.semilogy(Accsnnu, Xccsnnu, '*-',color = 'red', markersize=4, label='Mass no Fraction $CCSN_ps$')
#plt.semilogy(A, Xi, 'o-',color = 'blue' ,markersize=3,linewidth=3 ,label='neutrino interaction on')
#plt.semilogy(Aj, Xj, 's-',color = 'red', markersize=6, label='Mass Fraction $X_Low$')
#plt.semilogy(Al, Xl, 'd-',color = 'red', markersize=5, label='Mass Fraction $X_Scaled Down$')
#plt.semilogy(Ah, Xh, '^-',color = 'blue', m1arkersize=4, label='Mass Fraction $X_Scaled Up$')
#plt.semilogy(Aof, Xof, 'v-',color = 'red', markersize=3,linewidth=3 ,label='neutrino interaction off')
#plt.semilogy(Accsn, Xccsn, 'p-', markersize=2, label='Mass Fraction $CCSN_ps$')
#plt.semilogy(Aneuoff, Xneuoff, 'o-',color = 'red' ,markersize=4, label='Mass Fraction $neuoff$')
#plt.semilogy(Accsnnup, Xccsnnup, '*-',color = 'orange', markersize=4, label='Mass Fraction $CCSN_neup$')
#plt.semilogy(AMRSN, XMRSN, 'o-',color = 'blue' ,markersize=3,linewidth=2 ,label='neutrino interaction on')
#plt.semilogy(AnoMRSN_niOff, XnoMRSN_niOff, 'v-',color = 'red', markersize=3,linewidth=2 ,label='nNi57 (p,g) rate /1000')
plt.semilogy(AnoCCSN_ni, XnoCCSN_ni, '*-',color = 'black', markersize=5,linewidth=2 ,label='CCSN nNi57 (p,g) norm')
#plt.semilogy(AnoCCSN_nix100, XnoCCSN_nix100, 'o-',color = 'blue', markersize=3,linewidth=2 ,label='CCSN Ni57 (p,g) rate x100')
#plt.semilogy(AnoCCSN_nid100, XnoCCSN_nid100, 'o-',color = 'red', markersize=3,linewidth=2 ,label='CCSN Ni57 (p,g) rate /100')


#plt.semilogy(common_A,delta_X, 'v-',color = 'black', markersize=3,linewidth=2 ,label='nNi57 (p,g) rate /1000')
#plt.plot(common_A, np.abs(delta_X), 'v-',color = 'black', markersize=3,linewidth=2 ,label='nNi57 (p,g) rate /1000')

plt.fill_between(common_A,delta_X1,delta_X2, alpha=0.3, color='red', label='Reaclib Error Region')
#plt.fill_between(Temp, total2 - t2eL, total2 + t2eH, alpha=0.3, color='blue', label='Statistical Error')

plt.tick_params(axis='both', labelsize=22)
plt.xlim(15, 125)

plt.xlabel("Mass Number $A$", fontsize=24)
plt.ylabel("Abundance Fraction ratio (rxn x100 / div 100) ", fontsize=18)
plt.title("Final Mass Fractions ratio from Winnet Output", fontsize=26)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.legend(fontsize=22)
plt.legend(prop={'size': 25})
plt.legend()
plt.tight_layout()
plt.show()
