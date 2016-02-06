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

import pandas as pd
import numpy as np

import gpmcc.utils.config as cu
import gpmcc.utils.data as du
from gpmcc.engine import Engine

print 'Loading dataset ...'
df = pd.read_csv('malawi.csv')
df.replace('NA', np.nan, inplace=True)

schema = [
    ('sampleid','ignore'), ('site','categorical'), ('familyid','ignore'),
    ('personid','ignore'), ('zygosity','bernoulli'),
    ('twinpairphenotype','bernoulli'), ('healthcategory','bernoulli'),
    ('gender','bernoulli'), ('agemonths','normal'), ('whz','normal'),
    ('haz','normal'), ('waz','normal'), ('muac','normal'),
    ('fever','normal'), ('cough','normal'), ('diarrhea','normal'),
    ('vomiting','normal'), ('rutfx','categorical'), ('month','ignore'),
    ('year','ignore'), ('numberofhighqualitydeprelicatedh','normal'),
    ('humansequencesremoved','normal'),
    ('numberofv416srrnasequences','normal'),
    ('v416srrnasequencingrunid3','ignore'), ('acetobacteraceae_uncl','normal'),
    ('acidaminococcaceae_uncl','normal'), ('acidaminococcus','normal'),
    ('akkermansia','normal'), ('alistipes','normal'), ('alkaliphilus','normal'),
    ('anaerococcus','normal'), ('atopobium','normal'), ('bacteroides','normal'),
    ('bifidobacterium','normal'), ('blautia','normal'),
    ('brachyspira','normal'), ('butyrivibrio','normal'),
    ('campylobacter','normal'), ('catenibacterium','normal'),
    ('chlamydiaceae_uncl','normal'), ('citrobacter','normal'),
    ('clostridiales_family_xi_incertae_sedis_uncl','normal'),
    ('clostridium','normal'), ('collinsella','normal'),
    ('coprobacillus','normal'), ('coprococcus','normal'),
    ('desulfobulbaceae_uncl','normal'), ('desulfovibrio','normal'),
    ('dorea','normal'), ('enterobacter','normal'), ('enterococcus','normal'),
    ('escherichia','normal'), ('eubacterium','normal'),
    ('faecalibacterium','normal'), ('frankia','normal'),
    ('fusobacterium','normal'), ('gardnerella','normal'),
    ('haemophilus','normal'), ('helicobacter','normal'),
    ('klebsiella','normal'), ('lactobacillus','normal'),
    ('lactococcus','normal'), ('leuconostoc','normal'), ('megamonas','normal'),
    ('megasphaera','normal'), ('methanobrevibacter','normal'),
    ('micrococcus','normal'), ('mitsuokella','normal'),
    ('mycobacterium','normal'), ('neisseria','normal'),
    ('neisseriaceae_uncl','normal'), ('odoribacter','normal'),
    ('olsenella','normal'), ('parabacteroides','normal'),
    ('peptoniphilus','normal'), ('phascolarctobacterium','normal'),
    ('prevotella','normal'), ('providencia','normal'), ('pseudomonas','normal'),
    ('roseburia','normal'), ('rothia','normal'), ('ruminococcus','normal'),
    ('shigella','normal'), ('slackia','normal'),
    ('sphingobacteriaceae_uncl','normal'), ('staphylococcus','normal'),
    ('streptococcus','normal'), ('subdoligranulum','normal'),
    ('sutterella','normal'), ('veillonella','normal'), ('victivallis','normal'),
    ('unclassified','normal')
]

print 'Parsing schema ...'
T, cctypes, distargs, valmap, columns = du.parse_schema(schema, df)

print 'Initializing engine ...'
engine = Engine(T, cctypes, distargs=distargs, num_states=48, initialize=1)

print 'Analyzing for 28800 seconds (8 hours) ...'
engine.transition(S=28800, multithread=1)

print 'Pickling ...'
engine.to_pickle(file('%s-malawi.engine' % cu.timestamp(), 'w'))
