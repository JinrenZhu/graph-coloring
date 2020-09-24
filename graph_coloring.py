# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
import numpy as np
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

# Graph coloring with DQM solver

# input: number of colors in the graph
num_colors = 4
colors = np.linspace(0, num_colors - 1, num_colors)

# Initialize the DQM object
dqm = DiscreteQuadraticModel()

# Make Networkx graph of a hexagon
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)])
n_edges = len(G)

# Load the DQM. Define the variables, and then set quadratic weights.
# No biases necessary because we don't care what the colors are, as long as
# they are different at the edges.
for p in G.nodes:
    dqm.add_variable(4, label=p)
for p0, p1 in G.edges:
    dqm.set_quadratic(p0, p1, {(c, c): 1 for c in colors})

# Initialize the DQM solver
sampler = LeapHybridDQMSampler(profile='dqm_test')

# Solve the problem using the DQM solver
sampleset = sampler.sample_dqm(dqm)

# get the first solution, and print it
sample = sampleset.first.sample
energy = sampleset.first.energy
print(sample, energy)

# check that colors are different
valid = True
for edge in G.edges:
    i, j = edge
    if sample[i] == sample[j]:
        valid = False
        break
print("Graph coloring solution: ", sample)
print("Graph coloring solution energy: ", energy)
print("Graph coloring solution validity: ", valid)
