# coding: utf-8

import numpy as np
import random
import os
import pickle
from scipy.special import digamma
import time
import pandas as pd
import argparse

np.random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file_sbs', type=str, required=True)
args = parser.parse_args()

sbs96_df = pd.read_csv(args.input_file_sbs, sep='\t', index_col=0)

sbs_names = []
for i in range(0, 96):
    sbs_names.append(sbs96_df.columns[i][0]+sbs96_df.columns[i][2]+sbs96_df.columns[i][4]+sbs96_df.columns[i][6])

total_mutation_num = sbs96_df.values.shape[0]
train_num = int(0.7*total_mutation_num)
test_num = total_mutation_num-train_num

shuffle_idx = np.random.permutation(total_mutation_num)

train_mutation = sbs96_df.values[shuffle_idx[0:train_num],:]
test_mutation = sbs96_df.values[shuffle_idx[train_num:],:]

class flda():
    def __init__(self, factors,         sigmaA0=1.0, sigmaAB0=1.0, sigmaW0=0.5, sigmaWB0=10.0,         stepSizeADZ0=1e-2, stepSizeAZ0=1e-4, stepSizeAB0=1e-4, stepSizeW0=1e-3, stepSizeWB0=1e-5, stepSizeB0=1e-3,         delta00=0.1, delta10=0.1, alphaB0=-5.0, omegaB0=-5.0, likelihoodFreq0=100, blockFreq0=0):
        
        self.factor_num = len(factors)
        self.factors = factors.copy()
        self.factors_sub = factors.copy()
        
        self.factor_total = 1
        for i in range(self.factor_num-1, -1, -1):
            self.factors_sub[i] = self.factor_total
            self.factor_total *= self.factors[i]
            
        self.iToVector = {}
        for i in range(0, self.factor_total):
            self.iToVector[i] = self.iToVector_init(i)
            
        self.sigmaA = sigmaA0
        self.sigmaAB = sigmaAB0
        self.sigmaW = sigmaW0
        self.sigmaWB = sigmaWB0
        self.stepSizeADZ = stepSizeADZ0
        self.stepSizeAZ = stepSizeAZ0
        self.stepSizeAB = stepSizeAB0
        self.stepSizeW = stepSizeW0
        self.stepSizeWB = stepSizeWB0
        self.stepSizeB = stepSizeB0
        self.delta0 = delta00
        self.delta1 = delta10
        self.alphaB = alphaB0
        self.omegaB = omegaB0
        
        self.likelihoodFreq = likelihoodFreq0
        self.blockFreq = blockFreq0
        
    def iToVector_init(self, x):
        z = []
        
        for i in range(0, self.factor_num):
            z.append(int(x/self.factors_sub[i]))
            x = x%self.factors_sub[i]
            
        return z
    
    def parameter_init(self):
        print('Parameter initialize!')
        self.alphaZ = []
        self.alphaDZ = []
        
        for i in range(0, self.factor_num):
            self.alphaZ.append(np.zeros(self.factors[i]))
            self.alphaDZ.append(np.zeros((self.factors[i], self.doc_num)))
    
        self.priorDZ = np.zeros((self.doc_num, self.factor_total))
        self.alphaNorm = np.zeros(self.doc_num)
        
        self.omegaW = np.zeros(self.mutation_num)
        self.omegaZW = []
        self.priorZW = np.zeros((self.factor_total, self.mutation_num))
        self.omegaNorm = np.zeros(self.factor_total)
        
        for i in range(0, self.factor_num):
            self.omegaZW.append(np.zeros((self.factors[i], self.mutation_num)))
        
        self.beta = np.zeros(self.factor_total)
        
        self.nDZ = np.zeros((self.doc_num, self.factor_total)).astype(np.int32)
        self.nD = np.zeros(self.doc_num).astype(np.int32)
        self.nZW = np.zeros((self.factor_total, self.mutation_num)).astype(np.int32)
        self.nZ = np.zeros((self.factor_total))
        
        for i in range(0, self.mutation_num):
            self.omegaW[i] = self.etaW[i]
            
        for i in range(0, self.factor_num):
            for z in range(0, self.factors[i]):
                for w in range(0, self.mutation_num):
                    self.omegaZW[i][z,w] = self.etaZW[i][z,w]
                    
        for i in range(0, self.factor_total):
            for w in range(0, self.mutation_num):
                self.priorZW[i, w] = self.priorW(w, i)
                self.omegaNorm[i] += self.priorZW[i, w]
                
        for d in range(0, self.doc_num):
            for i in range(0, self.factor_total):
                self.priorDZ[d, i] = self.priorA(d, i)
                self.alphaNorm[d] += self.priorDZ[d, i]
                
        print('Frist sampling!')
        self.word_sampling = []
        
        start = time.time()
        for w in range(0, self.mutation_num):
            prob = self.priorZW[:, w]/np.sum(self.priorZW[:, w])
            
            self.word_sampling.append(list(np.random.choice(self.factor_total, size = self.mutations_count[w],replace = True , p = prob)))
        
        self.docsZ = np.zeros((self.doc_num, self.factor_total, self.mutation_num))
        
        start = time.time()
        for d in range(0, self.doc_num):
            self.nD[d] = len(self.docs[d])
            
            for w in range(0, self.mutation_num):
                
                if self.mutations[d,w] == 0:
                    continue
                
                cur_word_num = self.mutations[d,w]
                cur_word_factor = self.word_sampling[w][:cur_word_num]
                unique_factor, counts = np.unique(cur_word_factor, return_counts=True)
                self.docsZ[d,unique_factor, w] = counts
                
                self.word_sampling[w] = self.word_sampling[w][cur_word_num:]

        self.nZW = np.sum(self.docsZ, axis=0)
        self.nZ = np.sum(self.docsZ, axis=(0,2))
        self.nDZ = np.sum(self.docsZ, axis=(2))
            
    def priorA(self, d, x):
        weight = self.alphaB
        
        z = self.iToVector[x]
        
        for i in range(0, self.factor_num):
            weight += self.alphaZ[i][z[i]] + self.alphaDZ[i][z[i]][d]
            
        b = self.logistic(self.beta[x])
        
        return b*np.exp(weight)
    
    
    def test_priorA(self, d, x):
        weight = self.alphaB
        
        z = self.iToVector[x]
        
        for i in range(0, self.factor_num):
            weight += self.alphaZ[i][z[i]] + self.test_alphaDZ[i][z[i]][d]
            
        b = self.logistic(self.beta[x])
        
        return b*np.exp(weight)
    
    
    def priorW(self, w, x):
        weight = self.omegaB + self.omegaW[w]
        
        z = self.iToVector[x]
        
        for i in range(0, self.factor_num):
            weight += self.omegaZW[i][z[i], w]
            
        return np.exp(weight)
    
    def logistic(self, x):
        return 1.0 / (1.0 + np.exp(-1.0*x))
    
    def dlogistic(self, x):
        return self.logistic(x) * (1.0 - self.logistic(x))
    
    def mutations2docs(self, input_mutations, mutations_name):
        
        self.docs = []
        
        self.mutations_name = mutations_name
        
        self.doc_num = input_mutations.shape[0]
        self.mutation_num = input_mutations.shape[1]
        
        for i in range(0, self.doc_num):
            temp_doc = []
            for sbs in range(0, self.mutation_num):
                for count in range(0, int(input_mutations[i][sbs])):
                    temp_doc.append(sbs)
            
            self.docs.append(temp_doc)
        
        self.mutations = input_mutations.astype(np.int32)
        self.mutations_count = np.sum(input_mutations, axis = 0).astype(np.int32)
            
        self.etaW = np.zeros(self.mutation_num)
        self.etaZW = []
        
        for i in range(0, self.factor_num):
            self.etaZW.append(np.zeros((self.factors[i], self.mutation_num)))
            
    def mutations2docs_test_and_initialize(self, input_mutations):
        self.test_docs = []
        
        self.test_doc_num = input_mutations.shape[0]
        
        for i in range(0, self.test_doc_num):
            temp_doc = []
            for sbs in range(0, self.mutation_num):
                for count in range(0, int(input_mutations[i][sbs])):
                    temp_doc.append(sbs)
            
            self.test_docs.append(temp_doc)
        
        self.test_mutations = input_mutations.astype(np.int32)
        self.test_mutations_count = np.sum(input_mutations, axis = 0).astype(np.int32)
        
        self.test_alphaDZ = []
        
        for i in range(0, self.factor_num):
            self.test_alphaDZ.append(np.zeros((self.factors[i], self.test_doc_num)))
        
        self.test_priorDZ = np.zeros((self.test_doc_num, self.factor_total))
        self.test_alphaNorm = np.zeros(self.test_doc_num)
        
        for d in range(0, self.test_doc_num):
            for i in range(0, self.factor_total):
                self.test_priorDZ[d, i] = self.test_priorA(d, i)
                self.test_alphaNorm[d] += self.test_priorDZ[d, i]
                
        self.test_nDZ = np.zeros((self.test_doc_num, self.factor_total)).astype(np.int32)
        self.test_nD = np.zeros(self.test_doc_num).astype(np.int32)
#         self.nZW = np.zeros((self.factor_total, self.mutation_num)).astype(np.int32)
#         self.nZ = np.zeros((self.factor_total))
                
        print('Test sampling!')
        self.word_sampling = []
        
        start = time.time()
        for w in range(0, self.mutation_num):
            prob = self.priorZW[:, w]/np.sum(self.priorZW[:, w])
            
            self.word_sampling.append(list(np.random.choice(self.factor_total, size = self.test_mutations_count[w], replace = True , p = prob)))
        
        self.test_docsZ = np.zeros((self.test_doc_num, self.factor_total, self.mutation_num))
        
        start = time.time()
        for d in range(0, self.test_doc_num):
            self.test_nD[d] = len(self.test_docs[d])
            
            for w in range(0, self.mutation_num):
                
                if self.test_mutations[d,w] == 0:
                    continue
                
                cur_word_num = self.test_mutations[d,w]
                cur_word_factor = self.word_sampling[w][:cur_word_num]
                unique_factor, counts = np.unique(cur_word_factor, return_counts=True)
                self.test_docsZ[d,unique_factor, w] = counts
                
                self.word_sampling[w] = self.word_sampling[w][cur_word_num:]

#         self.test_nZW = np.sum(self.docsZ, axis=0)
#         self.test_nZ = np.sum(self.docsZ, axis=(0,2))
        self.test_nDZ = np.sum(self.test_docsZ, axis=(2))
            
    def updateWeights(self, iteration):
        self.updateWeightsW(iteration)
        self.updateWeightsA(iteration)
        
    def updateWeightsW(self, iteration):
        if iteration <= 20:
            return
        
        sigma = self.sigmaW
        
        gradientB = 0
        gradientW = np.zeros(self.mutation_num)
        
        dg1 = digamma(self.omegaNorm + 1e-8)
        dg2 = digamma(self.omegaNorm +self.nZ + 1e-8)
        
        dgW1 = digamma(self.priorZW + self.nZW + 1e-8)
        dgW2 = digamma(self.priorZW + 1e-8)
        
        gradientLL = self.priorZW*(np.expand_dims(dg1-dg2,1)+dgW1-dgW2)
        
        gradientZW = []
    
        
        for i in range(0, self.factor_num):
            gradientZW.append(np.zeros((self.factors[i], self.mutation_num)))
            
        for x in range(0, self.factor_total):
            z = self.iToVector[x]
            
            for i in range(0, self.factor_num):
                gradientZW[i][z[i],:] += gradientLL[x,:]
                
        gradientW += np.sum(gradientLL, 0)
        gradientB += np.sum(gradientLL)
        
        for i in range(0, self.factor_num):
            gradientZW[i] += -(self.omegaZW[i] - self.etaZW[i])/(sigma*sigma)
            self.omegaZW[i] += self.stepSizeW * gradientZW[i]
        
        gradientW += -(self.omegaW - self.etaW)/(sigma*sigma)
        self.omegaW = self.omegaW + (self.stepSizeW)*gradientW
            
        gradientB += -self.omegaB/(self.sigmaWB*self.sigmaWB)
        self.omegaB = self.omegaB + self.stepSizeWB*gradientB
        
    def updateWeightsA(self, iteration):
        sigma = self.sigmaA
        
        gradientBeta = np.zeros(self.factor_total)
        gradientB = 0
        
        gradientZ = []
        
        for i in range(0, self.factor_num):
            gradientZ.append(np.zeros(self.factors[i]))
            
        gradientDZ = []
        
        for i in range(0, self.factor_num):
            gradientDZ.append(np.zeros((self.factors[i], self.doc_num)))
        
        dg1 = digamma(self.alphaNorm + 1e-8)
        dg2 = digamma(self.alphaNorm + self.nD + 1e-8)
        dgW1 = digamma(self.priorDZ + self.nDZ + 1e-8)
        dgW2 = digamma(self.priorDZ + 1e-8)
        
        gradientLL = self.priorDZ*(np.expand_dims(dg1-dg2,1)+dgW1-dgW2)
        gradientB += np.sum(gradientLL)
        gradientBeta += np.sum(gradientLL * (1.0 - np.expand_dims(self.logistic(self.beta), 0)))
        
        for x in range(0, self.factor_total):
            z = self.iToVector[x]
            
            for i in range(0, self.factor_num):
                gradientZ[i][z[i]] += np.sum(gradientLL[:,x])
                gradientDZ[i][z[i],:] += gradientLL[:,x]
        
        for i in range(0, self.factor_num):
            gradientDZ[i] += - self.alphaDZ[i] / (sigma*sigma)
            self.alphaDZ[i] += self.stepSizeADZ*gradientDZ[i]
            
            gradientZ[i] += -self.alphaZ[i] / (sigma*sigma)
            self.alphaZ[i] += self.stepSizeAZ*gradientZ[i]
            
        gradientB += -self.alphaB/(sigma*sigma)
        self.alphaB += self.stepSizeAB*gradientB
        
        if iteration <= 20:
            return
    
        gradientBeta += (self.delta0 - 1.0) * self.dlogistic(self.beta)/self.logistic(self.beta)
        gradientBeta += (self.delta1 - 1.0) * (-1.0*self.dlogistic(self.beta)) / (1.0-self.logistic(self.beta))
            
        self.beta += self.stepSizeB*gradientBeta
        
    def test_updateWeightsA(self, iteration):
        sigma = self.sigmaA
        
#         gradientBeta = np.zeros(self.factor_total)
#         gradientB = 0
        
#         gradientZ = []
        
#         for i in range(0, self.factor_num):
#             gradientZ.append(np.zeros(self.factors[i]))
            
        gradientDZ = []
        
        for i in range(0, self.factor_num):
            gradientDZ.append(np.zeros((self.factors[i], self.test_doc_num)))
        
        dg1 = digamma(self.test_alphaNorm + 1e-8)
        dg2 = digamma(self.test_alphaNorm + self.test_nD + 1e-8)
        dgW1 = digamma(self.test_priorDZ + self.test_nDZ + 1e-8)
        dgW2 = digamma(self.test_priorDZ + 1e-8)
        
        gradientLL = self.test_priorDZ*(np.expand_dims(dg1-dg2,1)+dgW1-dgW2)
#         gradientB += np.sum(gradientLL)
#         gradientBeta += np.sum(gradientLL * (1.0 - np.expand_dims(self.logistic(self.beta), 0)))
        
        for x in range(0, self.factor_total):
            z = self.iToVector[x]
            
            for i in range(0, self.factor_num):
#                 gradientZ[i][z[i]] += np.sum(gradientLL[:,x])
                gradientDZ[i][z[i],:] += gradientLL[:,x]
        
        for i in range(0, self.factor_num):
            gradientDZ[i] += - self.test_alphaDZ[i] / (sigma*sigma)
            self.test_alphaDZ[i] += self.stepSizeADZ*gradientDZ[i]
            
#             gradientZ[i] += -self.alphaZ[i] / (sigma*sigma)
#             self.alphaZ[i] += self.stepSizeAZ*gradientZ[i]
            
#         gradientB += -self.alphaB/(sigma*sigma)
#         self.alphaB += self.stepSizeAB*gradientB
        
#         if iteration <= 20:
#             return
    
#         gradientBeta += (self.delta0 - 1.0) * self.dlogistic(self.beta)/self.logistic(self.beta)
#         gradientBeta += (self.delta1 - 1.0) * (-1.0*self.dlogistic(self.beta)) / (1.0-self.logistic(self.beta))
            
#         self.beta += self.stepSizeB*gradientBeta
        
    def computeLL(self):
        LL = 0.0
        
        tokenLL = np.einsum('ij,jk->ijk', (self.nDZ + self.priorDZ) / np.expand_dims(self.nD + self.alphaNorm, 1), (self.nZW + self.priorZW) / np.expand_dims(self.nZ + self.omegaNorm, 1))
        
        tokenLL *= self.docsZ
        
        tokenLL = tokenLL[np.nonzero(tokenLL)]
        
        LL = np.sum(np.log(tokenLL+1e-6))
        
                
        return LL
    
    def test_compute_LL(self):
        LL = 0.0
        
        tokenLL = np.einsum('ij,jk->ijk', (self.test_nDZ + self.test_priorDZ) / np.expand_dims(self.test_nD + self.test_alphaNorm, 1), (self.nZW + self.priorZW) / np.expand_dims(self.nZ + self.omegaNorm, 1))
        
        tokenLL *= self.test_docsZ
        
        tokenLL = tokenLL[np.nonzero(tokenLL)]
        
        LL = np.sum(np.log(tokenLL+1e-6))
                
        return LL
    
    def test_compute_perplexity(self):
        LL = 0.0
        
        tokenLL = np.einsum('ij,jk->ijk', (self.test_nDZ + self.test_priorDZ) / np.expand_dims(self.test_nD + self.test_alphaNorm, 1), (self.nZW + self.priorZW) / np.expand_dims(self.nZ + self.omegaNorm, 1))
        
        tokenLL *= self.test_docsZ
        
        tokenLL = tokenLL[np.nonzero(tokenLL)]
        
        LL = np.sum(np.log(tokenLL+1e-6))
        
        perplexity = np.exp(-LL/np.sum(self.test_mutations_count))
                
        return perplexity
    
    def doSampling(self, iteration):

        if iteration == 0:
            pass
        else:
        
            self.nDZ = np.zeros((self.doc_num, self.factor_total)).astype(np.int32)
            self.nZW = np.zeros((self.factor_total, self.mutation_num)).astype(np.int32)
            self.nZ = np.zeros((self.factor_total))
            
            self.word_sampling = []
            
            start = time.time()
            for w in range(0, self.mutation_num):
                prob = self.priorZW[:, w]/np.sum(self.priorZW[:, w])
                
                self.word_sampling.append(list(np.random.choice(self.factor_total, size = self.mutations_count[w],replace = True , p = prob)))
            
            self.docsZ = np.zeros((self.doc_num, self.factor_total, self.mutation_num))
            
            
            for d in range(0, self.doc_num):
                
                for w in range(0, self.mutation_num):
                    
                    if self.mutations[d,w] == 0:
                        continue
                    
                    cur_word_num = self.mutations[d,w]
                    cur_word_factor = self.word_sampling[w][:cur_word_num]
                    unique_factor, counts = np.unique(cur_word_factor, return_counts=True)
                    self.docsZ[d,unique_factor,w] = counts
                    
                    self.word_sampling[w] = self.word_sampling[w][cur_word_num:]

            self.nZW = np.sum(self.docsZ, axis=0)
            self.nZ = np.sum(self.docsZ, axis=(0,2))
            self.nDZ = np.sum(self.docsZ, axis=(2))

        self.updateWeights(iteration)
        
        
        self.alphaNorm = np.zeros(self.doc_num)
        self.omegaNorm = np.zeros(self.factor_total)
        
        for i in range(0, self.factor_total):
            for w in range(0, self.mutation_num):
                self.priorZW[i, w] = self.priorW(w, i)
                self.omegaNorm[i] += self.priorZW[i, w]
                
        for d in range(0, self.doc_num):
            for i in range(0, self.factor_total):
                self.priorDZ[d, i] = self.priorA(d, i)
                self.alphaNorm[d] += self.priorDZ[d, i]
            
        LL = self.computeLL()
        print("Iter:%d Log-likelihood: %f"%(iteration,LL))
            
    def testUpdate(self, iteration):
        
        if iteration == 0:
            pass
        else:
            self.word_sampling = []
        
            for w in range(0, self.mutation_num):
                prob = self.priorZW[:, w]/np.sum(self.priorZW[:, w])

                self.word_sampling.append(list(np.random.choice(self.factor_total, size = self.test_mutations_count[w], replace = True , p = prob)))

            self.test_docsZ = np.zeros((self.test_doc_num, self.factor_total, self.mutation_num))

            start = time.time()
            for d in range(0, self.test_doc_num):

                for w in range(0, self.mutation_num):

                    if self.test_mutations[d,w] == 0:
                        continue

                    cur_word_num = self.test_mutations[d,w]
                    cur_word_factor = self.word_sampling[w][:cur_word_num]
                    unique_factor, counts = np.unique(cur_word_factor, return_counts=True)
                    self.test_docsZ[d,unique_factor, w] = counts

                    self.word_sampling[w] = self.word_sampling[w][cur_word_num:]
            self.test_nDZ = np.sum(self.test_docsZ, axis=(2))
        
        self.test_updateWeightsA(iteration)
        
        
        self.test_alphaNorm = np.zeros(self.test_doc_num)
        
        for d in range(0, self.test_doc_num):
            for i in range(0, self.factor_total):
                self.test_priorDZ[d, i] = self.test_priorA(d, i)
                self.test_alphaNorm[d] += self.test_priorDZ[d, i]
        
        perplexity = self.test_compute_perplexity()
    
        print("Iter:%d Test perplexity: %f"%(iteration, perplexity))


flda_model = flda([26])
flda_model.mutations2docs(train_mutation, sbs_names)
flda_model.parameter_init()

for i in range(0, 250+1):
    flda_model.doSampling(i)
    if i % 50 == 0:
        bytes_out = pickle.dumps(flda_model)
        file_path = "output/storage_perplexity_baseline_"+str(i)+".pkl"
        max_bytes = 2**31 - 1
        with open(file_path, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])
