import numpy as np
import matplotlib .pyplot as plt
import random
import math
#real number problem(integer)
# here we use permutation encoding
#we use order 1 base crossover
#we use swap mutation
#we use roul wheel selection
#maximize fitness function [1/(1+clashes)]  when the fitness =1 is the best it means that there is no clashes "clashes=0"
N_queen = int(input("enter number of queens: ")) # the user can enter any integer number of queens
init_num_sol=100 #parameter
ngen=100 #parameter
pcross=0.8#parameter
pmute=0.1#parameter
new_generation=np.zeros((init_num_sol,N_queen))
bestHist = np.zeros(ngen)
#This function takes the size of population as the parameter and returns a list of chromosomes that contain randomly generated genes 
#we use random.sample bec we do not want duplicates 
def init_sol(init_num_sol,N_queen): #intialization of intial number solutions (population) and the solution represent an individual
  pop=np.zeros((init_num_sol,N_queen))
  for i in range(init_num_sol):
    chromo=random.sample(range(0,N_queen),N_queen)
    pop[i,:]=chromo
  return pop # return intial population of solutions
#------------------------------------------------------------------------------------------------------------
#clashes-----fitness (maximizeze)
# our objective function is 1/(1+clashes) so when you have zero clashes you get the maximum objective which is equal 1
def fit_func(chromosome): # calculate the fitness of the solution
  clashes=0
# calculate row and column clashes
# just subtract the unique length of array from total length of array
# [1,1,1,2,2,2] - [1,2] => 4 clashes
  row_col_clashes = abs(len(chromosome) - len(np.unique(chromosome)))
  clashes += row_col_clashes
  #print(clashes)
  
# calculate diagonal clashes

  for i in range(N_queen):
    for j in range (i+1,N_queen):
      if ( i != j):
        dx = abs(chromosome[i] - chromosome[j])
        dy = abs(i-j) 
        if(dx == dy):
          clashes += 1

  func=1/(1+clashes)
  return func #return the fitness of each individual
# to try if the function is running correctly or not  
#m=fit_func([0,1,2,3,6,5,4,7])
#m=fit_func([1,4,7,5,2,0,6,3])
#m=fit_func([5,1,2,7,6,3,0,4])
#print(m)  
#------------------------------------------------------------------------------------------------------------------
def calc_fit(pop,init_num_sol): #calculate the fntness of each solution
    pop_fit=np.zeros(init_num_sol)
    for i in range (len(pop)):
        f=fit_func(pop[i]) #number
        pop_fit[i]=f
    return pop_fit  #return array of fitness of each solution   

#-------------------------------------------------------------------------------------------------------------------
def selection_prob(pop_fitness):
  total_fitness = sum(pop_fitness)
  selection_probs = pop_fitness/total_fitness
  return selection_probs
#-------------------------------------------------------------------------------------------------------------------
def cummProb(selection_probs):
  cumm_probs = np.zeros_like(selection_probs)
  cumm_probs[0] = selection_probs[0]
  for i in range(1,len(selection_probs)):
    cumm_probs[i] = cumm_probs[i-1] + selection_probs[i]
  return cumm_probs #return array of cummulative probability
#-------------------------------------------------------------------------------------------------------------------
def roulette_wheel(cumm_probs): 
  r = np.random.random()
  for i in range(len(cumm_probs)):
    if r <= cumm_probs[i]:
      indx = i
      return indx #index of the selected chromosome
#-------------------------------------------------------------------------------------------------------------------
def roulette_selection(cumm_probs, pop): 
  twoParents = np.zeros((2,np.size(pop,1)))
  for i in range(2):
    indx = roulette_wheel(cumm_probs)
    twoParents[i,:] = pop[indx,:]
  return twoParents
#-------------------------------------------------------------------------------------------------------------------
def Cross(twoParents,pcross,N_queen): #order base crossover
  twoChildren=np.zeros((2,N_queen))
  x=np.random.random()
  if x<=pcross:
    crossPoint1=np.random.randint(1,N_queen)
    crossPoint2=np.random.randint(1,N_queen)
    while (crossPoint1 >= crossPoint2):
      crossPoint1=np.random.randint(1,N_queen)
      crossPoint2=np.random.randint(1,N_queen)  
    for l in range(2):
      arr1=np.zeros(N_queen-(crossPoint2-crossPoint1))#remained
      arr2=np.zeros(crossPoint2-crossPoint1)#interval between cross points
      m=0
      for i in range(crossPoint1,crossPoint2):
        arr2[m]=twoParents[l][i]
        m=m+1
      j=0
      for i in range(N_queen):
        if(l==0):
          if(twoParents[l+1][i] not in arr2 ):
            arr1[j]=twoParents[l+1][i]
            j=j+1
        if(l==1):
          if(twoParents[l-1][i] not in arr2 ):
            arr1[j]=twoParents[l-1][i]
            j=j+1


      twoChildren[l,:]=np.hstack((arr1[:crossPoint1],arr2[:],arr1[crossPoint1:]))
  else:
     twoChildren=twoParents   
  return twoChildren
#u=binCross([[0,5,7,1,6,3,2,4],[6,2,0,1,4,7,3,5]],0.99,8)
#print(u)
#--------------------------------------------------------------------------------------------------------------------  
def mutate(individual,pmute,N_queen):# swap mutaion
  mutatedInd=individual[:]
  if np.random.random()<pmute:
      rand1=np.random.randint(0,N_queen)
      rand2=np.random.randint(0,N_queen)
      while (rand1== rand2):
        rand1=np.random.randint(0,N_queen)
        rand2=np.random.randint(0,N_queen)
      #print(rand1)
      #print(rand2)
      mutatedInd[rand1], mutatedInd[rand2] = mutatedInd[rand2], mutatedInd[rand1]  
  return mutatedInd 
#r=binmutate([2, 1, 7, 4, 6, 5, 0, 3],0.1,8) 
#print(r)
#---------------------------------------------------------------------------------------------------------------------

def runGA(init_num_sol,N_queen,ngen,pcross,pmute):
    pop = init_sol(init_num_sol,N_queen)
    for z in range(ngen):
        pop_fitness = calc_fit(pop,init_num_sol)
        prop_fits = selection_prob(pop_fitness)
        cumm_probs = cummProb(prop_fits)

        for counter in range(int(init_num_sol/2)):
            selected_index = roulette_wheel(cumm_probs)
            twoParents = roulette_selection(cumm_probs, pop)
            twoChildren = Cross(twoParents, pcross, N_queen)

            for x in range(2):
                for y in range(N_queen):
                    new_generation[2 * counter + x][y] = twoChildren[x][y]
        for i in range(len(new_generation)):
            mutated_ind = mutate(new_generation[i], pmute, N_queen)
            new_generation[i] = mutated_ind
        pop=new_generation
        pop_fitness = calc_fit(pop,init_num_sol)
        maxElement = np.amax(pop_fitness)
        bestHist[z] = maxElement
        
    return new_generation,bestHist
new_generation,bestHist=runGA(init_num_sol,N_queen,ngen,pcross,pmute)
print(bestHist)



plt.plot(bestHist)
plt.xlabel("gen")
plt.ylabel("fitness")
plt.show()

