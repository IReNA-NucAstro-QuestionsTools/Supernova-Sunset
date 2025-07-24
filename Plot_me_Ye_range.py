# Author: M. Reichert
#edited by Sean P Byrne

import numpy as np
import matplotlib.pyplot as plt


def extract_first_last_ye(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Remove comment lines and empty lines
    data_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

    if not data_lines:
        raise ValueError("No data lines found in file.")

    # Extract Ye values (5th column, index 4)
    first_ye = float(data_lines[0].split()[4])
    last_ye = float(data_lines[-1].split()[4])

    return first_ye, last_ye

if __name__ == "__main__":
    filepath = "mainout.dat"
    first_ye, last_ye = extract_first_last_ye(filepath)
    print(f"First Ye: {first_ye:.4f}")
    print(f"Last Ye:  {last_ye:.4f}")




fig, ax = plt.subplots(7,1,figsize=(5,5),sharex=True)
plt.subplots_adjust(hspace=0)
time,temp,rho,Ye,Yn,Yp,Yhvy,ent = np.loadtxt("mainout.dat",unpack=True,usecols=[1,2,3,4,6,8,10,13])

# WinNet density and temperature
ax[0].plot(time-time[0],rho/1e6,color="tab:blue",lw=4,alpha=0.7,label="Density")
ax[1].plot(time-time[0],temp,color="tab:red",lw=4,alpha=0.7,label="temp")
ax[2].plot(time-time[0],Ye,color="tab:green",lw=4,alpha=0.7,label="Ye")
ax[3].plot(time-time[0],Yn,color="tab:red",lw=4,alpha=0.7,label="Yn")
ax[4].plot(time-time[0],Yp,color="tab:blue",lw=4,alpha=0.7,label="Yp")
ax[5].plot(time-time[0],ent,color="tab:green",lw=4,alpha=0.7,label="entropy")
ax[6].plot(time-time[0],Yhvy,color="tab:green",lw=4,alpha=0.7,label="Yhvy")

# Calculate the thermodynamic quantities and check if they are okay:
R0    = 0.2
rho_0 = 1e6
T9_analytic  = lambda x: 2.4*(R0)**(-3./4.)*np.exp(-x/ (3*(446/np.sqrt(7*rho_0))))
rho_analytic = lambda x: 7*rho_0 *np.exp(-x / (446/np.sqrt(7*rho_0)))
T9_gridpoint  = T9_analytic(time)
rho_gridpoint = rho_analytic(time)
ax[0].plot(time-time[0],rho_gridpoint/1e6,ls="--",color="k",label="Analytic")
ax[1].plot(time-time[0],T9_gridpoint,ls="--",color="k")


fig2 = plt.figure(figsize=(5,3))
ax2  = fig2.gca()
A,X  = np.loadtxt("finabsum.dat",unpack=True,usecols=[0,2])
ax2.plot(A,X)
ax2.set_xlim(0,80)
ax2.set_ylim(1e-8,1)
ax2.set_yscale("log")
ax2.set_title("Final mass fractions")
ax2.set_ylabel("Mass fraction X")
ax2.set_xlabel("Mass number A")


ax[0].set_ylabel(r"$\rho$ [10$^6$ g cm$^{-3}$]")
ax[1].set_ylabel("T[GK]")
ax[2].set_xlabel("Time [s]")
ax[3].set_xlim(0,3)
ax[4].legend()
ax[5].legend()
ax[6].legend()


fig2.savefig("final_massfractions.pdf",bbox_inches="tight")

fig3 = plt.figure(figsize=(5,3))
ax3 = fig3.gca()
T,Y  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
T1,Y1  = np.loadtxt("../CCSN_winds_Ye_9_F3/mainout.dat",unpack=True,usecols=[1,4])
T2,Y2  = np.loadtxt("../CCSN_winds_Ye_8_F3/mainout.dat",unpack=True,usecols=[1,4])
T3,Y3  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
T4,Y4  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
TE,YE  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])

ax3.plot(T,Y)
ax3.plot(T1,Y1)
ax3.plot(T2,Y2)
ax3.plot(T3,Y3)
ax3.plot(T4,Y4)
ax3.plot(TE,YE)
ax3.set_xlim(0.01,0.5)
#ax3.set_yscale('log')


fig4 = plt.figure(figsize=(5,3))
ax4 = fig4.gca()
Tc,Yc  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
Tc1,Yc1  = np.loadtxt("../CCSN_winds_Ye_9_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc2,Yc2  = np.loadtxt("../CCSN_winds_Ye_8_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc3,Yc3  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
Tc4,Yc4  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
TcE,YcE  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])

Ycr=Yc-0.4
Yc1r=Yc1-0.4
Yc2r=Yc2-0.3

ax4.plot(Tc,Ycr)
ax4.plot(Tc1,Yc1r)
ax4.plot(Tc2,Yc2r)
#ax4.plot(Tc3,Yc3)
#ax4.plot(Tc4,Yc4)
#ax4.plot(TcE,YcE)
ax4.set_xlim(0.01,0.5)
#ax3.set_yscale('log')

if __name__ == "__main__":
    filepath = "../CCSN_winds_Ye_9/mainout.dat"
    first_ye94, last_ye94 = extract_first_last_ye(filepath)
    print(f" Ye: 0.90 flag4")
    print(f"First Ye: {first_ye94:.4f}")
    print(f"Last Ye:  {last_ye94:.4f}")


if __name__ == "__main__":
    filepath = "../CCSN_winds_Ye_9_F3/mainout.dat"
    first_ye93, last_ye93 = extract_first_last_ye(filepath)
    print(f" Ye: 0.90 Flag 3")
    print(f"First Y: {first_ye93:.4f}")
    print(f"Last Ye:  {last_ye93:.4f}")

if __name__ == "__main__":
    filepath = "../CCSN_winds_Ye_8_F3/mainout.dat"
    first_ye80, last_ye80 = extract_first_last_ye(filepath)
    print(f" Ye: 0.80")
    print(f"First Ye: {first_ye80:.4f}")
    print(f"Last Ye:  {last_ye80:.4f}")

if __name__ == "__main__":
    filepath = "../Working_CCSN_explosive_burning_parametrized_neup_on2/mainout.dat"
    first_ye40, last_ye40 = extract_first_last_ye(filepath)
    print(f" Ye: 0.4")
    print(f"First Ye: {first_ye40:.4f}")
    print(f"Last Ye:  {last_ye40:.4f}")







plt.show()
