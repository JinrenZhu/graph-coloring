# %% pre
from dwave.system import DWaveSampler, EmbeddingComposite

# %% test
sampler = DWaveSampler()
print(sampler.properties)