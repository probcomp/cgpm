# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2014 Baxter S. Eaves Jr,
# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Lead Developer: Feras Saad <fsaad@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

_colors = ['black', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
    'pink']

_plot_layout = {
    1: (1,1),
    2: (2,1),
    3: (3,1),
    4: (2,2),
    5: (3,2),
    6: (3,2),
    7: (4,2),
    8: (4,2),
    9: (3,3),
    10: (5,2),
    11: (4,3),
    12: (4,3),
    13: (5,3),
    14: (5,3),
    15: (5,3),
    16: (4,4),
    17: (6,3),
    18: (6,3),
    19: (5,4),
    20: (5,4),
    21: (7,3),
    22: (6,4),
    23: (6,4),
    24: (6,4),
}

def get_state_plot_layout(n_cols):
    layout = dict()
    layout['plots_x'] = _plot_layout[n_cols][0]
    layout['plots_y'] = _plot_layout[n_cols][1]
    layout['plot_inches_x'] = 13/6. * layout['plots_x']
    layout['plot_inches_y'] = 6. * layout['plots_y']
    layout['border_color'] = _colors
    return layout
