import networkx as nx
import numpy as np
import os
import geopandas as gpd
import random
import sys

import districtingGraph
import initializer
import multiLayeredGraph
import multiScaleMergeSplit
import metropolisHastings
# import dataWriter

from importlib import reload

reload(districtingGraph)
reload(initializer)
reload(multiLayeredGraph)
reload(multiScaleMergeSplit)
reload(metropolisHastings)

print('initializing run...')
state, info = initializer.setRunParametersFromCommandLine(sys.argv)
proposal, info = multiScaleMergeSplit.define(info)
info = initializer.fillMissingInfoFields(info)
state = initializer.determineStateInfo(state, info)

heirarchy = {1: 'County', 0: 'VTD'}
state = multiLayeredGraph.buildGraph(state, heirarchy, info['initPlan'])
state = multiLayeredGraph.buildDistricting(state, 1, state["layeredGraph"]["graphLevels"][1].nodes)

print('run initialized...')

reload(metropolisHastings)
reload(multiScaleMergeSplit)

print('starting chain...')
state = metropolisHastings.run(state, proposal, info)
print('finishing chain...')