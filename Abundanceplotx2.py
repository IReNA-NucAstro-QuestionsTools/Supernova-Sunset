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
plt.semilogy(Aneuoff, Xneuoff, 'o-',color = 'red' ,markersize=4, linewidth = 3, label='Mass Fraction neutino interaction off')
#plt.semilogy(Accsnnup, Xccsnnup, '*-',color = 'orange', markersize=4, label='Mass Fraction $CCSN_neup$')
plt.semilogy(AMRSN, XMRSN, 'o-',color = 'blue' ,markersize=3,linewidth=3 ,label='neutrino interaction on')
#plt.semilogy(AnoMRSN_niOff, XnoMRSN_niOff, 'v-',color = 'red', markersize=3,linewidth=2 ,label='nNi57 (p,g) rate /1000')

plt.tick_params(axis='both', labelsize=22)
plt.xlim(45, 125)

plt.xlabel("Mass Number $A$", fontsize=24)
plt.ylabel("Abundance Fraction ", fontsize=24)
plt.title("Final Mass Fractions from Winnet Output", fontsize=26)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.legend(fontsize=22)
plt.legend(prop={'size': 25})
plt.legend()
plt.tight_layout()
plt.show()
