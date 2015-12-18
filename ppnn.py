import numpy as np
import random

VECTOR_SIZE = 10

sequence_codes = {}
char_codes = {}

class Neural_Network(object):
    def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = VECTOR_SIZE
        self.outputLayerSize = 3
        self.hiddenLayerSize = 20
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
        
## ----------------------- Part 6 ---------------------------- ##
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 1000, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

'''
return vector of given length containing random values between 0 and 1
'''
def rand_vector(length):
    return [random.random() for i in range(length)]

def char_code_vector(window):
    coded_vector = [0,0,0,0,0]
    count = 0
    w = window[:]
    for c in w:
        if c in char_codes:
            coded_vector[count] = char_codes[c]
        else:
            char_codes[c] = random.random()
            coded_vector[count] = char_codes[c]
    count += 1
    return coded_vector

'''
convert secondary structures into integers - 0 for helixes, 1 for beta sheets, 2 for coils(or unknown values)
'''
def secondary_structure_code(structure):
    if structure in ('G', 'H', 'I', 'T'):
        return [1,0,0]
    elif structure in ('E', 'B'):
        return [0,1,0]
    else:#most likely 'C', 'S', ' ', or '-', barring error in file
        return [0,0,1]

def dssp_parser(filename):
    protein = ""
    secondary_structures = ""
    pro_array = []
    sec_array = []
    pro_seq = {}
    sequence = 1 #indicates whether current sequence in file is protein or secondary structures
    #count = 0

    with open(filename, "r") as fo:
        for line in fo:
            if line[0] != '>':
                if sequence == 0:
                    protein += line.rstrip('\n')
                else:
                    secondary_structures += line.rstrip('\n')
            elif sequence == 0:
                sequence = 1
            else:
                sequence = 0
                #count += 1
                for index in range(len(protein)):
                    window = ""
                    if index == 0:
                        window = '--' + protein[:index + 3]
                    elif index == 1:
                        window = '-' + protein[:index + 3]
                    elif index == len(protein) - 2:
                        window = protein[index - 2:] + '-'
                    elif index == len(protein) - 1:
                        window = protein[index - 2:] + '--'
                    else:
                        window = protein[index - 2:index + 3]
                    #print protein
                    #print secondary_structures
                    #if index < len(secondary_structures):
                    pro_array.append(window)
                    
                    if window not in sequence_codes:
                        sequence_codes[window] = char_code_vector(window) + rand_vector(VECTOR_SIZE/2)
                    pro_seq[window] = sequence_codes[window]
                    
                    #pro_seq[window] = char_code_vector(window) #rand_vector(VECTOR_SIZE) #
                    sec_array.append(secondary_structures[index])
                protein = ""
                secondary_structures = ""
    amino_acid_codes = []
    secondary_structures = []
    for i in range(len(pro_array)):
       amino_acid_codes.append(pro_seq[pro_array[i]])
       secondary_structures.append(secondary_structure_code(sec_array[i]))

    a = np.array(amino_acid_codes)
    s = np.array(secondary_structures)
    #a = a/np.amax(a, axis=0)
    return a, s
'''
def nn_dict_converter(pro_array, sec_array):
    amino_acid_codes = []
    secondary_structures = []
    for i in range(len(pro_array)):
       amino_acid_codes.append(rand_vector(VECTOR_SIZE))
       secondary_structures.append([secondary_structure_code(value)])

    a = np.array(amino_acid_codes)
    s = np.array(secondary_structures)
    #a = a/np.amax(a, axis=0)
    return a, s
'''
def classify(yHat):
    classified = []
    for i in yHat:
        if i <= 0.366666667:
            classified.append(0.1)
        elif i <= 0.633333333:
            classified.append(0.5)
        else:
            classified.append(0.9)
    return classified

def accuracy_checker(output, real_values):
    correct = 0.0
    total = 0.0
    for i in range(len(output)):
        if output[i] == real_values[i]:
            correct += 1
        total += 1
    return correct / total

