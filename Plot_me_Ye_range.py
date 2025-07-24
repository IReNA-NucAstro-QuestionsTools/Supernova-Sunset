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
A,X  = np.loadtxt("../CCSN_winds_Ye_9/finabsum.dat",unpack=True,usecols=[0,2])
A1,X1  = np.loadtxt("../CCSN_winds_Ye_9_F3/finabsum.dat",unpack=True,usecols=[0,2])
A2,X2  = np.loadtxt("../CCSN_winds_Ye_8_F3/finabsum.dat",unpack=True,usecols=[0,2])
A3,X3  = np.loadtxt("../CCSN_winds_Ye_7_F3/finabsum.dat",unpack=True,usecols=[0,2])
A4,X4  = np.loadtxt("../CCSN_winds_Ye_65_F3/finabsum.dat",unpack=True,usecols=[0,2])
AE,XE  = np.loadtxt("../CCSN_winds_Ye_9/finabsum.dat",unpack=True,usecols=[0,2])






ax2.plot(A,X)
ax2.plot(A1,X1)
ax2.plot(A2,X2)
ax2.plot(A3,X3)
ax2.plot(A4,X4)
ax2.plot(AE,XE)



ax2.set_xlim(0,300)
ax2.set_ylim(1e-8,1)
ax2.set_yscale("log")
ax2.set_title("Final mass fractions")
ax2.set_ylabel("Mass fraction X")
ax2.set_xlabel("Mass number A")


ax[0].set_ylabel(r"$\rho$ [10$^6$ g cm$^{-3}$]")
ax[1].set_ylabel("T[GK]")
ax[2].set_xlabel("Time [s]")
ax[3].set_xlim(0,3)
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()
ax[5].legend()
ax[6].legend()


fig2.savefig("final_massfractions.pdf",bbox_inches="tight")

fig3 = plt.figure(figsize=(5,3))
ax3 = fig3.gca()
T,Y  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
T1,Y1  = np.loadtxt("../CCSN_winds_Ye_9_F3/mainout.dat",unpack=True,usecols=[1,4])
T2,Y2  = np.loadtxt("../CCSN_winds_Ye_8_F3/mainout.dat",unpack=True,usecols=[1,4])
T3,Y3  = np.loadtxt("../CCSN_winds_Ye_7_F3/mainout.dat",unpack=True,usecols=[1,4])
T4,Y4  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T5,Y5  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T6,Y6  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T7,Y7  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T8,Y8  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T9,Y9  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T10,Y10  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T11,Y11  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
TE,YE  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])

ax3.plot(T,Y,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
ax3.plot(T1,Y1,color="blue",lw=4,alpha=0.7,label="Ye = .9_F3")
ax3.plot(T2,Y2,color="red",lw=4,alpha=0.7,label="Ye = .8")
ax3.plot(T3,Y3,color="green",lw=4,alpha=0.7,label="Ye = .7")
ax3.plot(T4,Y4,color="cyan",lw=4,alpha=0.7,label="Ye = .65")
ax3.plot(T5,Y5,color="orange",lw=4,alpha=0.7,label="Ye = .6")
ax3.plot(T6,Y6,color="pink",lw=4,alpha=0.7,label="Ye = .55")
ax3.plot(T7,Y7,color="turquoise",lw=4,alpha=0.7,label="Ye = .5")
ax3.plot(T8,Y8,color="sea-green",lw=4,alpha=0.7,label="Ye = .45")
ax3.plot(T9,Y9,color="tan",lw=4,alpha=0.7,label="Ye = .4")
ax3.plot(T10,Y10,color="brown",lw=4,alpha=0.7,label="Ye = .3")
ax3.plot(T11,Y11,color="gold",lw=4,alpha=0.7,label="Ye = 0.2")
ax3.plot(TE,YE,color="grey",lw=4,alpha=0.7,label="Ye = .41")

ax3.set_xlim(0.01,0.5)
#ax3.set_yscale('log')
ax3.legend()

fig4 = plt.figure(figsize=(5,3))
ax4 = fig4.gca()
Tc,Yc  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
Tc1,Yc1  = np.loadtxt("../CCSN_winds_Ye_9_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc2,Yc2  = np.loadtxt("../CCSN_winds_Ye_8_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc3,Yc3  = np.loadtxt("../CCSN_winds_Ye_7_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc4,Yc4  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc5,Yc5  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc6,Yc6  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc7,Yc7  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc8,Yc8  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc9,Yc9  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc10,Yc10  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc11,Yc11  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
TcE,YcE  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])


# normalize the change to Ye to 0.5 to compare change
Ycr=Yc-0.4
Yc1r=Yc1-0.4
Yc2r=Yc2-0.3
Yc3r=Yc3-0.2
Yc4r=Yc4-0.15
Yc5r=Yc5-0.15
Yc6r=Yc6-0.15
Yc7r=Yc7-0.15
Yc8r=Yc8-0.15
Yc9r=Yc9-0.15
Yc10r=Yc10-0.15
Yc11r=Yc11-0.15
YcEr=YcE-0.15



ax4.plot(Tc,Ycr,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
ax4.plot(Tc1,Yc1r,color="blue",lw=4,alpha=0.7,label="Ye = .9_F3")
ax4.plot(Tc2,Yc2r,color="red",lw=4,alpha=0.7,label="Ye = .8_F3")
ax4.plot(Tc3,Yc3r,color="green",lw=4,alpha=0.7,label="Ye = .7_F3")
ax4.plot(Tc4,Yc4r,color="cyan",lw=4,alpha=0.7,label="Ye = .65_F3")
ax4.plot(Tc5,Yc5r,color="orange",lw=4,alpha=0.7,label="Ye = .6_F3")
ax4.plot(Tc6,Yc6r,color="pink",lw=4,alpha=0.7,label="Ye = .55_F3")
ax4.plot(Tc7,Yc7r,color="turquoise",lw=4,alpha=0.7,label="Ye = .5_F3")
ax4.plot(Tc8,Yc8r,color="sea-green",lw=4,alpha=0.7,label="Ye = .45_F3")
ax4.plot(Tc9,Yc9r,color="tan",lw=4,alpha=0.7,label="Ye = .4_F3")
ax4.plot(Tc10,Yc10r,color="brown",lw=4,alpha=0.7,label="Ye = .3_F3")
ax4.plot(Tc11,Yc11r,color="gold",lw=4,alpha=0.7,label="Ye = .2_F3")
ax4.plot(Tc12,Yc12r,color="grey",lw=4,alpha=0.7,label="Ye = .41_F3")
#ax4.plot(TcE,YcE,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
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
    filepath = "../CCSN_winds_Ye_7_F3/mainout.dat"
    first_ye7, last_ye7 = extract_first_last_ye(filepath)
    print(f" Ye: 0.7")
    print(f"First Ye: {first_ye7:.4f}")
    print(f"Last Ye:  {last_ye7:.4f}")

if __name__ == "__main__":
    filepath = "../CCSN_winds_Ye_65_F3/mainout.dat"
    first_ye65, last_ye65 = extract_first_last_ye(filepath)
    print(f" Ye: 0.65")
    print(f"First Ye: {first_ye65:.4f}")
    print(f"Last Ye:  {last_ye65:.4f}")







plt.show()
