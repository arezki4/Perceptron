from pylab import *
import numpy as np
import warnings
import csv


warnings.filterwarnings('ignore')


"""
IMPORT DES DONNEES A PARTIR DE BEBE.CSV
"""
with open('bebe.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    next(spamreader) #skip la première ligne
    S = []
    N = []
    P = []
    for row in spamreader:
    	arr = row[0].split(";")
    	S.append(arr[0])
    	N.append(float(arr[1]))
    	P.append(float(arr[2])/10000)
    	#print(row)

"""
NORMALISATION DES DONNEES AVANT TRAITEMENT
"""
for i in range(len(S)):
	if S[i]=='M':
		S[i] = 0
	else:
		S[i] = 1
       
     
"""
S = H/F
N = nb semaine
P = poid des bébés
w = les poids
b = biais
"""     
       
b = 0.5 #biais

w1=random()
w2=random()
w3=random()
w4=random()
w5=random()
w6=random()

w7=random()
w8=random()
w9=random()

w10=random()
w11=random()
w12=random()

w13=random()
w14=random()
w15=random()

w16=random()
w17=random()
w18=random()

w19=random()
w20=random()
w21=random()

w22=random()
w23=random()
w24=random()

w25=random()
w26=random()
w27=random()



w_first = ([w1,w2,w3,w4,w7,w8]) #poids à modifier
w_second = ([w5,w6,w9]) 

couche = [0] * len(w_second) #Initialisation de la couche cachée

def sigmoid(x):
	return 1/(1+np.exp(-x))

def tangenteHyper(x):
	return (1-np.exp(-2*x))/(1+np.exp(-2*x))
	
k = 0 #nombre d'itération du réseau afin de l'affiner

while(k<400):#Limitation du nombre d'iteration de l'apprentissage
	somme = 0
	#DE INPUTS A COUCHE CACHEE
	for i in range(350): #ICI S ET N SONT A 0 POUR LES PREMIERS TESTS MAIS IL FAUDRA RÉGLER TOUT ÇA
		for j in range(0,len(w_first),2):
			couche[round(j/2)] = sigmoid(S[i]*w_first[j] + N[i]*w_first[j+1])
		
		#DE COUCHE CACHEE A OUTPUT
		output = 0
		for j in range(0,len(w_second),1):
			output += couche[j]*w_second[j]
		output = sigmoid(output)
		
		#APRÈS AVOIR TROUVÉ L'OUTPUT FINALE, IL FAUT VOIR SI NOTRE RÉSULTAT EST LE BON, SI OUI ON PASSE A LA DONNEE SUIVANTE, SINON, ON RETROPROPAGE L'ERREUR
		
		delta = P[i]-output
		print("Output : "+str(output)+" P["+str(i)+"] = "+str(P[i])+" || différence = "+str(delta))
				
		if ((delta > 0.1) or (delta < -0.1)): #Si la valeur déduite est trop éloignée, retropapger
			couche_delta = [0] * len(w_second)
			for j in range(0,len(couche),1):
				couche_delta[j] = 1 - (couche[j]**2) * w_second[j] * delta #Dérivé partielle
				#couche_delta[j] = couche[j] * (1-couche[j]) * w_second[j] * delta #Dérivé partielle			
			#ARRIVÉ ICI, IL FAUT MAINTENANT METTRE À JOUR LES POIDS
			for j in range(0,len(w_first),2):
				w_first[j] = w_first[j] + b * S[i] * couche_delta[round(j/2)]
				w_first[j+1] = w_first[j+1] + b * N[i] * couche_delta[round(j/2)]
			for j in range(0,len(w_second),1):
				w_second[j] = w_second[j] + b * couche[j] * delta
			somme += (100 - (abs(P[i]-output)/abs(P[i]))) / 100
	k +=1
	b = (somme/350) * output * (1-output)
	
#ICI LA PHASE D'APPRENTISSAGE EST TERMINÉE, IL FAUT DONC MAINTENANT DÉDUIRE LES RÉSULTATS DE 351 À 500

print("\n\n+==================================TEST OUTPUT=================================+\n\n")

for i in range(147): #ICI S ET N SONT A 0 POUR LES PREMIERS TESTS MAIS IL FAUDRA RÉGLER TOUT ÇA
		for j in range(0,len(w_first),2):
			couche[round(j/2)] = sigmoid(S[i+350]*w_first[j] + N[i+350]*w_first[j+1])
		
		#DE COUCHE CACHEE A OUTPUT
		output = 0
		for j in range(0,len(w_second),1):
			output += couche[j]*w_second[j]
		output = sigmoid(output)
		
		#APRÈS AVOIR TROUVÉ L'OUTPUT FINALE, IL FAUT VOIR SI NOTRE RÉSULTAT EST LE BON, SI OUI ON PASSE A LA DONNEE SUIVANTE, SINON, ON RETROPROPAGE L'ERREUR
		
		delta = P[i+350]-output
		
		print("test-Output : "+str(output)+" P["+str(i+350)+"] = "+str(P[i+350])+" || différence = "+str(delta))


	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
