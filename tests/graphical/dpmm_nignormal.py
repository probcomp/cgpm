# -*- coding: utf-8 -*-

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import matplotlib.pyplot as plt

from cgpm.crosscat.state import State

def observe_datum(x):
    global state
    state.incorporate(rowid=state.n_rows(), observation={0:x})
    state.transition_dim_grids()
    print 'Observation %d: %f' % (state.n_rows(), x)
    while True:
        state.transition_view_rows()
        state.transition_dim_hypers()
        state.transition_view_alphas()
        ax.clear()
        state.dim_for(0).plot_dist(
            state.X[0], Y=np.linspace(0.01,0.99,200), ax=ax)
        ax.grid()
        plt.pause(.8)

def on_click(event):
    if event.button == 1:
        if event.inaxes is not None:
            observe_datum(event.xdata)

# Create state.
initial_point = .8
state = State([[initial_point]], cctypes=['normal'])

# Activate plotter.
fig, ax = plt.subplots()
ax.grid()
plt.connect('button_press_event', on_click)
plt.show()
