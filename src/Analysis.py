import pandas as pd
import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.svm import SVC

import argparse

from flda_model import flda

parser = argparse.ArgumentParser()
parser.add_argument('--input_file_sbs', type=str, required=True)
parser.add_argument('--input_file_sample', type=str, required=True)
parser.add_argument('--input_file_clinical', type=str, required=True)
parser.add_argument('--input_file_flda_result', type=str, required=True)
parser.add_argument('--input_file_flda_baseline_result', type=str, required=True)
parser.add_argument('--input_file_supp', type=str, required=True)
parser.add_argument('--input_file_supp_table_name', type=str, required=True)
parser.add_argument('--output_fig_1', type=str, required=True)
parser.add_argument('--output_fig_2', type=str, required=True)
parser.add_argument('--output_fig_3', type=str, required=True)
parser.add_argument('--output_fig_4', type=str, required=True)
args = parser.parse_args()

np.random.seed(123)

df = pd.read_excel(io=args.input_file_supp, sheet_name=args.input_file_supp_table_name)

sig = df.values[1:,2:7].astype('float')


max_bytes = 2**31 - 1
bytes_in = bytearray(0)
input_size = os.path.getsize(args.input_file_flda_result)
with open(args.input_file_flda_result, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
        
flda_model = pickle.loads(bytes_in)

priorZW = flda_model.priorZW
probZW = priorZW/np.sum(priorZW, axis = 1, keepdims=True)

_ = plt.figure()
_ = sns.heatmap(1.-cdist(sig.transpose(), probZW, metric='cosine'), annot=False)
_ = plt.xlabel('Inferred Signatures by fLDA')
_ = plt.ylabel('Signatures of Alexandrov2016')

plt.savefig(args.output_fig_1)


_ = plt.figure()
_ = sns.heatmap(1.-cdist(probZW, probZW, metric='cosine'), annot=False)
_ = plt.xlabel('Inferred Signatures by fLDA')
_ = plt.ylabel('Inferred Signatures by fLDA')

plt.savefig(args.output_fig_2)


max_bytes = 2**31 - 1
bytes_in = bytearray(0)
input_size = os.path.getsize(args.input_file_flda_baseline_result)
with open(args.input_file_flda_baseline_result, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
        
flda_baseline_model = pickle.loads(bytes_in)

priorZW_baseline = flda_baseline_model.priorZW
probZW_baseline= priorZW_baseline/np.sum(priorZW_baseline, axis = 1, keepdims=True)

_ = plt.figure()
_ = sns.heatmap(1.-cdist(sig.transpose(), probZW_baseline, metric='cosine'), annot=False)
_ = plt.xlabel('Inferred Signatures by fLDA baseline')
_ = plt.ylabel('Signatures of Alexandrov2016')

plt.savefig(args.output_fig_3)

_ = plt.figure()
_ = sns.heatmap(1.-cdist(probZW_baseline, probZW_baseline, metric='cosine'), annot=False)
_ = plt.xlabel('Inferred Signatures by fLDA baseline')
_ = plt.ylabel('Inferred Signatures by fLDA baseline')

plt.savefig(args.output_fig_4)

sbs96_df = pd.read_csv(args.input_file_sbs, sep='\t', index_col=0)
sample_df = pd.read_csv(args.input_file_sample, sep='\t', index_col=0)
clinical_df = pd.read_csv(args.input_file_clinical, sep='\t', index_col=0)

sample2patient = {}
patient2smoke = {}

sample_list = list(sbs96_df.index)

sample_df_sample_name = list(sample_df.values[:,0])

sample_df_patient = list(sample_df.index)

for i in range(0, len(sample_df_sample_name)):
    sample2patient[sample_df_sample_name[i]] = sample_df_patient[i]

clinical_df_smoke = list(clinical_df.values[:,0])

clinical_df_patient = list(clinical_df.index)

for i in range(0, len(clinical_df_smoke)):
    if clinical_df_smoke[i] == 'Smoker':
        patient2smoke[clinical_df_patient[i]] = 1
    elif clinical_df_smoke[i] == 'Non-Smoker':
        patient2smoke[clinical_df_patient[i]] = -1
    elif math.isnan(clinical_df_smoke[i]):
        patient2smoke[clinical_df_patient[i]] = 0

sample2smoke = np.zeros(len(sample_list))

for i in range(0, len(sample_list)):
    if sample_list[i] not in sample2patient or sample2patient[sample_list[i]] not in patient2smoke:
        sample2smoke[i] = 0
    else:
        sample2smoke[i] = patient2smoke[sample2patient[sample_list[i]]]


sampleWithSmokeHistory = list(np.nonzero(sample2smoke)[0])


flda_exposure = flda_model.priorDZ
flda_exposure = flda_exposure/np.sum(flda_exposure, axis = 1, keepdims=True)

flda_baseline_exposure = flda_baseline_model.priorDZ
flda_baseline_exposure = flda_baseline_exposure/np.sum(flda_baseline_exposure, axis = 1, keepdims=True)

sbs96_ws = sbs96_df.values[sampleWithSmokeHistory,:]
flda_exposure_ws = flda_exposure[sampleWithSmokeHistory,:]
flda_baseline_exposure_ws = flda_baseline_exposure[sampleWithSmokeHistory,:]
sample2smoke_ws = sample2smoke[sampleWithSmokeHistory]


total_num = len(sampleWithSmokeHistory)

shuffle_idx = np.random.permutation(total_num)

train_num = int(0.7*total_num)
test_num = total_num-train_num

sbs96_ws_t = sbs96_ws[shuffle_idx[0:train_num],:]
sbs96_ws_v = sbs96_ws[shuffle_idx[train_num:],:]

flda_exposure_ws_t = flda_exposure_ws[shuffle_idx[0:train_num],:]
flda_exposure_ws_v = flda_exposure_ws[shuffle_idx[train_num:],:]

flda_baseline_exposure_ws_t = flda_baseline_exposure_ws[shuffle_idx[0:train_num],:]
flda_baseline_exposure_ws_v = flda_baseline_exposure_ws[shuffle_idx[train_num:],:]

sample2smoke_ws_t = sample2smoke_ws[shuffle_idx[0:train_num]]
sample2smoke_ws_v = sample2smoke_ws[shuffle_idx[train_num:]]


clf_1 = SVC(gamma='auto')
clf_1.fit(sbs96_ws_t, sample2smoke_ws_t)
print('Classification result from mutation: ', clf_1.score(sbs96_ws_v, sample2smoke_ws_v))


clf_2 = SVC(gamma='auto')
clf_2.fit(flda_exposure_ws_t, sample2smoke_ws_t)
print('Classification result from fLDA: ',clf_2.score(flda_exposure_ws_v, sample2smoke_ws_v))


clf_3 = SVC(gamma='auto')
clf_3.fit(flda_baseline_exposure_ws_t, sample2smoke_ws_t)
print('Classification result from fLDA baseline: ',clf_3.score(flda_baseline_exposure_ws_v, sample2smoke_ws_v))

clf_1 = SVC(gamma='auto',class_weight='balanced')
clf_1.fit(sbs96_ws_t, sample2smoke_ws_t)
print('Classification result from mutation (balanced mode): ', clf_1.score(sbs96_ws_v, sample2smoke_ws_v))


clf_2 = SVC(gamma='auto',class_weight='balanced')
clf_2.fit(flda_exposure_ws_t, sample2smoke_ws_t)
print('Classification result from fLDA (balanced mode): ',clf_2.score(flda_exposure_ws_v, sample2smoke_ws_v))


clf_3 = SVC(gamma='auto',class_weight='balanced')
clf_3.fit(flda_baseline_exposure_ws_t, sample2smoke_ws_t)
print('Classification result from fLDA baseline (balanced mode): ',clf_3.score(flda_baseline_exposure_ws_v, sample2smoke_ws_v))



