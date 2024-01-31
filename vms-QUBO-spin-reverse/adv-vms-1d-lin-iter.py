
# %% pre
import numpy as np

from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
from dwave.preprocessing import SpinReversalTransformComposite
from dwave.inspector import show

# %% parameters
nel = 8
nen = nel + 1
n_real = nen - 2
n_encode = 3
n_bin = n_real * n_encode
n_iter = 3

# Sampler
QA_n_reads = 4000
QA_sample_rtol = 1.0e-4
QA_chain_strength = 3
QA_time = 50

itest = 2

# suffix
sffx_1 = str(n_real).zfill(2)
sffx_2 = str(n_encode).zfill(2)
sffx_iter = str(n_iter).zfill(3)
sffx_3 = str(QA_n_reads).zfill(5)
sffx_4 = str(QA_chain_strength).zfill(4)
sffx_5 = str(QA_time).zfill(4)
sffx_6 = str(itest).zfill(2)

sffx_12i = sffx_1 + '_' + sffx_2 + '_' + sffx_iter + '.npy'
sffx_all = sffx_1 + '_' + sffx_2 + '_' + sffx_iter + '_' \
    + sffx_3 + '_' + sffx_4 + '_' + sffx_5 + '_' + sffx_6 + '.npy'

# %% load QUBO coefficients
cov_matrix_off = np.load('cov_mat_off_' + sffx_12i)
bias_linear = np.load('bias_linear_' + sffx_12i)

# %% sampling
# generate QUBO
bqm_adv = BinaryQuadraticModel('BINARY')
bqm_adv.add_linear_from_array(bias_linear)
bqm_adv.add_quadratic_from_dense(cov_matrix_off)

# samplerQA = EmbeddingComposite(SpinReversalTransformComposite(DWaveSampler()))
samplerQA = EmbeddingComposite(DWaveSampler())
sampleset_QA = samplerQA.sample(bqm_adv,
                                num_reads=QA_n_reads,
                                annealing_time=QA_time,
                                chain_strength=QA_chain_strength,
                                label='Adv-VMS-3bits-1')

print(sampleset_QA.first)
print(len(sampleset_QA.lowest(rtol=QA_sample_rtol))/len(sampleset_QA))

np.save('QA_samples_' + sffx_all, sampleset_QA.record)

show(sampleset_QA)