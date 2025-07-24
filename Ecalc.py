import numpy as np

Cross_sec_Ratio=8.35082e49
Cons_L=3e51/Cross_sec_Ratio
Cons_E=25.88*Cross_sec_Ratio

intervals=np.round(np.arange(0.20, 0.95, 0.05), 2)

for i in intervals:
	E=3e51/(((1/i)-1)*((2e51)/(16.66))/1.43756)
	L=25.88*(((1/i)-1)*((2e51)/(16.66))/1.43756)
	print("Constant Luminosity")
	print(i)
	print(E)
	#print(i)
	#print(L)
	#print("")
	#print(Cons_E)
	#print(Cons_L)
		



