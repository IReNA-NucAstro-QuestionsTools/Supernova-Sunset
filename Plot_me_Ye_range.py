# Author: M. Reichert
#edited by Sean P Byrne

import numpy as np
import matplotlib.pyplot as plt

#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
#quick function pulling the last and first Ye values to compare changes per file

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

#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
# pull the function from every run

if __name__ == "__main__":
    filepath = "mainout.dat"
    first_ye, last_ye = extract_first_last_ye(filepath)
    print(f"First Ye: {first_ye:.4f}")
    print(f"Last Ye:  {last_ye:.4f}")



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


if __name__ == "__main__":
    filepath = "../Working_CCSN_wind_bliss_0.60/mainout.dat"
    first_ye6, last_ye6 = extract_first_last_ye(filepath)
    print(f" Ye: 0.6")
    print(f"First Ye: {first_ye6:.4f}")
    print(f"Last Ye:  {last_ye6:.4f}")


if __name__ == "__main__":
    filepath = "../CCSN_wind_bliss_055/mainout.dat"
    first_ye55, last_ye55 = extract_first_last_ye(filepath)
    print(f" Ye: 0.55")
    print(f"First Ye: {first_ye55:.4f}")
    print(f"Last Ye:  {last_ye55:.4f}")


if __name__ == "__main__":
    filepath = "../CCSN_wind_bliss_050/mainout.dat"
    first_ye5, last_ye5 = extract_first_last_ye(filepath)
    print(f" Ye: 0.5")
    print(f"First Ye: {first_ye5:.4f}")
    print(f"Last Ye:  {last_ye5:.4f}")


if __name__ == "__main__":
    filepath = "../CCSN_wind_bliss_045/mainout.dat"
    first_ye45, last_ye45 = extract_first_last_ye(filepath)
    print(f" Ye: 0.45")
    print(f"First Ye: {first_ye45:.4f}")
    print(f"Last Ye:  {last_ye45:.4f}")


if __name__ == "__main__":
    filepath = "../Working_CCSN_wind_bliss_0.40/mainout.dat"
    first_ye4, last_ye4 = extract_first_last_ye(filepath)
    print(f" Ye: 0.4")
    print(f"First Ye: {first_ye4:.4f}")
    print(f"Last Ye:  {last_ye4:.4f}")


if __name__ == "__main__":
    filepath = "../Working_CCSN_wind_bliss_0.30/mainout.dat"
    first_ye3, last_ye3 = extract_first_last_ye(filepath)
    print(f" Ye: 0.3")
    print(f"First Ye: {first_ye3:.4f}")
    print(f"Last Ye:  {last_ye3:.4f}")



if __name__ == "__main__":
    filepath = "../Working_CCSN_wind_bliss_0.20/mainout.dat"
    first_ye2, last_ye2 = extract_first_last_ye(filepath)
    print(f" Ye: 0.2")
    print(f"First Ye: {first_ye2:.4f}")
    print(f"Last Ye:  {last_ye2:.4f}")



if __name__ == "__main__":
    filepath = "../CCSN_winds_Ye_41_OG/mainout.dat"
    first_yeE, last_yeE = extract_first_last_ye(filepath)
    print(f" Ye: 0.41")
    print(f"First Ye: {first_yeE:.4f}")
    print(f"Last Ye:  {last_yeE:.4f}")

#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
#Begin the plotting
#This is just a plot of the different outputs in the mainout to see the system

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

#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Calculate the thermodynamic quantities and check if they are okay:
R0    = 0.2
rho_0 = 1e6
T9_analytic  = lambda x: 2.4*(R0)**(-3./4.)*np.exp(-x/ (3*(446/np.sqrt(7*rho_0))))
rho_analytic = lambda x: 7*rho_0 *np.exp(-x / (446/np.sqrt(7*rho_0)))
T9_gridpoint  = T9_analytic(time)
rho_gridpoint = rho_analytic(time)
ax[0].plot(time-time[0],rho_gridpoint/1e6,ls="--",color="k",label="Analytic")
ax[1].plot(time-time[0],T9_gridpoint,ls="--",color="k")



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


#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
#This is the Abundance plot of all the simulations ran

fig2 = plt.figure(figsize=(5,3))
ax2  = fig2.gca()
A,X  = np.loadtxt("../CCSN_winds_Ye_9/finabsum.dat",unpack=True,usecols=[0,2])
A1,X1  = np.loadtxt("../CCSN_winds_Ye_9_F3/finabsum.dat",unpack=True,usecols=[0,2])
A2,X2  = np.loadtxt("../CCSN_winds_Ye_8_F3/finabsum.dat",unpack=True,usecols=[0,2])
A3,X3  = np.loadtxt("../CCSN_winds_Ye_7_F3/finabsum.dat",unpack=True,usecols=[0,2])
A4,X4  = np.loadtxt("../CCSN_winds_Ye_65_F3/finabsum.dat",unpack=True,usecols=[0,2])
A5,X5  = np.loadtxt("../Working_CCSN_wind_bliss_0.60/finabsum.dat",unpack=True,usecols=[0,2])
A6,X6  = np.loadtxt("../CCSN_wind_bliss_055/finabsum.dat",unpack=True,usecols=[0,2])
A7,X7  = np.loadtxt("../CCSN_wind_bliss_050/finabsum.dat",unpack=True,usecols=[0,2])
A8,X8  = np.loadtxt("../CCSN_wind_bliss_045/finabsum.dat",unpack=True,usecols=[0,2])
A9,X9  = np.loadtxt("../Working_CCSN_wind_bliss_0.40/finabsum.dat",unpack=True,usecols=[0,2])
A10,X10  = np.loadtxt("../Working_CCSN_wind_bliss_0.30/finabsum.dat",unpack=True,usecols=[0,2])
A11,X11  = np.loadtxt("../Working_CCSN_wind_bliss_0.20/finabsum.dat",unpack=True,usecols=[0,2])
AE,XE  = np.loadtxt("../CCSN_winds_Ye_41_OG/finabsum.dat",unpack=True,usecols=[0,2])




ax2.plot(A,X,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
ax2.plot(A1,X1,color="blue",lw=4,alpha=0.7,label="Ye = .9_F3")
ax2.plot(A2,X2,color="red",lw=4,alpha=0.7,label="Ye = .8")
ax2.plot(A3,X3,color="green",lw=4,alpha=0.7,label="Ye = .7")
ax2.plot(A4,X4,color="cyan",lw=4,alpha=0.7,label="Ye = .65")
ax2.plot(A5,X5,color="orange",lw=4,alpha=0.7,label="Ye = .6")
ax2.plot(A6,X6,color="pink",lw=4,alpha=0.7,label="Ye = .55")
ax2.plot(A7,X7,color="turquoise",lw=4,alpha=0.7,label="Ye = .5")
ax2.plot(A8,X8,color="seagreen",lw=4,alpha=0.7,label="Ye = .45")
ax2.plot(A9,X9,color="tan",lw=4,alpha=0.7,label="Ye = .4")
ax2.plot(A10,X10,color="brown",lw=4,alpha=0.7,label="Ye = .3")
ax2.plot(A11,X11,color="gold",lw=4,alpha=0.7,label="Ye = 0.2")
ax2.plot(AE,XE,color="grey",lw=4,alpha=0.7,label="Ye = .41")




ax2.set_xlim(0,300)
ax2.set_ylim(1e-8,1)
ax2.set_yscale("log")
ax2.set_title("Final mass fractions")
ax2.set_ylabel("Mass fraction X")
ax2.set_xlabel("Mass number A")
ax2.legend()


fig2.savefig("final_massfractions.pdf",bbox_inches="tight")

#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
#Plot with all the Ye's in the same plot

fig3 = plt.figure(figsize=(5,3))
ax3 = fig3.gca()
T,Y  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
T1,Y1  = np.loadtxt("../CCSN_winds_Ye_9_F3/mainout.dat",unpack=True,usecols=[1,4])
T2,Y2  = np.loadtxt("../CCSN_winds_Ye_8_F3/mainout.dat",unpack=True,usecols=[1,4])
T3,Y3  = np.loadtxt("../CCSN_winds_Ye_7_F3/mainout.dat",unpack=True,usecols=[1,4])
T4,Y4  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
T5,Y5  = np.loadtxt("../Working_CCSN_wind_bliss_0.60/mainout.dat",unpack=True,usecols=[1,4])
T6,Y6  = np.loadtxt("../CCSN_wind_bliss_055/mainout.dat",unpack=True,usecols=[1,4])
T7,Y7  = np.loadtxt("../CCSN_wind_bliss_050/mainout.dat",unpack=True,usecols=[1,4])
T8,Y8  = np.loadtxt("../CCSN_wind_bliss_045/mainout.dat",unpack=True,usecols=[1,4])
T9,Y9  = np.loadtxt("../Working_CCSN_wind_bliss_0.40/mainout.dat",unpack=True,usecols=[1,4])
T10,Y10  = np.loadtxt("../Working_CCSN_wind_bliss_0.30/mainout.dat",unpack=True,usecols=[1,4])
T11,Y11  = np.loadtxt("../Working_CCSN_wind_bliss_0.20/mainout.dat",unpack=True,usecols=[1,4])
TE,YE  = np.loadtxt("../CCSN_winds_Ye_41_OG/mainout.dat",unpack=True,usecols=[1,4])

ax3.plot(T,Y,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
ax3.plot(T1,Y1,color="blue",lw=4,alpha=0.7,label="Ye = .9_F3")
ax3.plot(T2,Y2,color="red",lw=4,alpha=0.7,label="Ye = .8")
ax3.plot(T3,Y3,color="seagreen",lw=4,alpha=0.7,label="Ye = .7")
ax3.plot(T4,Y4,color="cyan",lw=4,alpha=0.7,label="Ye = .65")
ax3.plot(T5,Y5,color="orange",lw=4,alpha=0.7,label="Ye = .6")
ax3.plot(T6,Y6,color="gold",lw=4,alpha=0.7,label="Ye = .55")
ax3.plot(T7,Y7,color="turquoise",lw=4,alpha=0.7,label="Ye = .5")
ax3.plot(T8,Y8,color="green",lw=4,alpha=0.7,label="Ye = .45")
ax3.plot(T9,Y9,color="tan",lw=4,alpha=0.7,label="Ye = .4")
ax3.plot(T10,Y10,color="brown",lw=4,alpha=0.7,label="Ye = .3")
ax3.plot(T11,Y11,color="pink",lw=4,alpha=0.7,label="Ye = 0.2")
ax3.plot(TE,YE,color="grey",lw=4,alpha=0.7,label="Ye = .41")

ax3.set_xlim(0.01,0.5)
#ax3.set_yscale('log')
#ax3.set_ylim(1e-8,1)
#ax3.set_yscale("log")
ax3.set_title("Change in Ye")
ax3.set_ylabel("Ye")
ax3.set_xlabel("Time")
ax3.legend()


#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
#Looking at the changes of all the Ye for the simulations but adjusted to start at 0.5 in order to directly compare
fig4 = plt.figure(figsize=(5,3))
ax4 = fig4.gca()
Tc,Yc  = np.loadtxt("../CCSN_winds_Ye_9/mainout.dat",unpack=True,usecols=[1,4])
Tc1,Yc1  = np.loadtxt("../CCSN_winds_Ye_9_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc2,Yc2  = np.loadtxt("../CCSN_winds_Ye_8_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc3,Yc3  = np.loadtxt("../CCSN_winds_Ye_7_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc4,Yc4  = np.loadtxt("../CCSN_winds_Ye_65_F3/mainout.dat",unpack=True,usecols=[1,4])
Tc5,Yc5  = np.loadtxt("../Working_CCSN_wind_bliss_0.60/mainout.dat",unpack=True,usecols=[1,4])
Tc6,Yc6  = np.loadtxt("../CCSN_wind_bliss_055/mainout.dat",unpack=True,usecols=[1,4])
Tc7,Yc7  = np.loadtxt("../CCSN_wind_bliss_050/mainout.dat",unpack=True,usecols=[1,4])
Tc8,Yc8  = np.loadtxt("../CCSN_wind_bliss_045/mainout.dat",unpack=True,usecols=[1,4])
Tc9,Yc9  = np.loadtxt("../Working_CCSN_wind_bliss_0.40/mainout.dat",unpack=True,usecols=[1,4])
Tc10,Yc10  = np.loadtxt("../Working_CCSN_wind_bliss_0.30/mainout.dat",unpack=True,usecols=[1,4])
Tc11,Yc11  = np.loadtxt("../Working_CCSN_wind_bliss_0.20/mainout.dat",unpack=True,usecols=[1,4])
TcE,YcE  = np.loadtxt("../CCSN_winds_Ye_41_OG/mainout.dat",unpack=True,usecols=[1,4])


# normalize the change to Ye to 0.5 to compare change
Ycr=Yc-0.4
Yc1r=Yc1-0.4
Yc2r=Yc2-0.3
Yc3r=Yc3-0.2
Yc4r=Yc4-0.15
Yc5r=Yc5-0.1
Yc6r=Yc6-0.05
Yc7r=Yc7
Yc8r=Yc8+0.05
Yc9r=Yc9+0.1
Yc10r=Yc10+0.2
Yc11r=Yc11+0.3
YcEr=YcE+0.09



ax4.plot(Tc,Ycr,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
ax4.plot(Tc1,Yc1r,color="blue",lw=4,alpha=0.7,label="Ye = .9_F3")
ax4.plot(Tc2,Yc2r,color="red",lw=4,alpha=0.7,label="Ye = .8_F3")
ax4.plot(Tc3,Yc3r,color="seagreen",lw=4,alpha=0.7,label="Ye = .7_F3")
ax4.plot(Tc4,Yc4r,color="cyan",lw=4,alpha=0.7,label="Ye = .65_F3")
ax4.plot(Tc5,Yc5r,color="orange",lw=4,alpha=0.7,label="Ye = .6_F3")
ax4.plot(Tc6,Yc6r,color="gold",lw=4,alpha=0.7,label="Ye = .55_F3")
ax4.plot(Tc7,Yc7r,color="turquoise",lw=4,alpha=0.7,label="Ye = .5_F3")
ax4.plot(Tc8,Yc8r,color="green",lw=4,alpha=0.7,label="Ye = .45_F3")
ax4.plot(Tc9,Yc9r,color="tan",lw=4,alpha=0.7,label="Ye = .4_F3")
ax4.plot(Tc10,Yc10r,color="brown",lw=4,alpha=0.7,label="Ye = .3_F3")
ax4.plot(Tc11,Yc11r,color="pink",lw=4,alpha=0.7,label="Ye = .2_F3")
ax4.plot(TcE,YcEr,color="grey",lw=4,alpha=0.7,label="Ye = .41_F3")
#ax4.plot(TcE,YcE,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
ax4.set_xlim(0.01,0.5)
#ax3.set_yscale('log')i
#ax4.set_ylim(1e-8,1)
#ax4.set_yscale("log")
ax4.set_title("Overlay of Ye changes")
ax4.set_ylabel("Ye trajectory but started to 0.5")
ax4.set_xlabel("Time")
ax4.legend()



#=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-
#Abundances of only the simulations in the ranfe of literature

fig5 = plt.figure(figsize=(5,3))
ax5 = fig5.gca()

#ax5.plot(A,X,color="black",lw=4,alpha=0.7,label="Ye = .9_F4")
#ax5.plot(A1,X1,color="blue",lw=4,alpha=0.7,label="Ye = .9_F3")
#ax5.plot(A2,X2,color="red",lw=4,alpha=0.7,label="Ye = .8")
#ax5.plot(A3,X3,color="seagreen",lw=4,alpha=0.7,label="Ye = .7")
ax5.plot(A4,X4,color="cyan",lw=4,alpha=0.7,label="Ye = .65")
ax5.plot(A5,X5,color="orange",lw=4,alpha=0.7,label="Ye = .6")
ax5.plot(A6,X6,color="gold",lw=4,alpha=0.7,label="Ye = .55")
ax5.plot(A7,X7,color="turquoise",lw=4,alpha=0.7,label="Ye = .5")
ax5.plot(A8,X8,color="green",lw=4,alpha=0.7,label="Ye = .45")
#ax5.plot(A9,X9,color="tan",lw=4,alpha=0.7,label="Ye = .4")
#ax5.plot(A10,X10,color="brown",lw=4,alpha=0.7,label="Ye = .3")
#ax5.plot(A11,X11,color="pink",lw=4,alpha=0.7,label="Ye = 0.2")
ax5.plot(AE,XE,color="grey",lw=4,alpha=0.7,label="Ye = .41")


ax5.set_xlim(0,150)
ax5.set_ylim(1e-8,1)
ax5.set_yscale("log")
ax5.set_title("Final mass fractions")
ax5.set_ylabel("Mass fraction X")
ax5.set_xlabel("Mass number A")
ax5.legend()


plt.show()
