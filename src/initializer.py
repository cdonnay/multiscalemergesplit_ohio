import os
import random

import districtingGraph

def setIdealPop(state, info):
    mean_huh = info["parameters"]["idealPop"] == "mean" 
    if mean_huh:
        graph = state["graph"]
        popTotal = sum([graph.nodes[n]["Population"] for n in graph.nodes])
        idealPop = popTotal/float(info["parameters"]["districts"])
        state["idealPop"] = idealPop
    else:
        state["idealPop"] = info["parameters"]["idealPop"]
    return state

def setPopBounds(state, info):
    try:
        popDiv = info["constraints"]["populationDeviation"]
        state["maxPop"] = state["idealPop"]*(1 + popDiv)
        state["minPop"] = state["idealPop"]*(1 - popDiv)
    except:
        graph = state["graph"]
        popTotal = sum([graph.nodes[n]["Population"] for n in graph.nodes])
        state["maxPop"] = popTotal
        state["minPop"] = 10.0**-14
    return state

def setPopulation(state, info):
    graph = state["graph"]    
    popTotal = sum([graph.nodes[n]["Population"] for n in graph.nodes])
    state["Population"] = popTotal
    return state

def determineStateInfo(state, info):
    state = setPopulation(state, info)
    state = setIdealPop(state, info)
    state = setPopBounds(state, info)
    state["maxCntySplts"] = info["parameters"]["maxSplitCounties"]
    return state

def fillMissingInfoFields(info):
    if "step" not in info["parameters"]:
        info["parameters"]["step"] = 0
    if "idealPop" not in info["parameters"]:
        info["parameters"]["idealPop"] = "mean"
    return info

def readCmd(cmd, val, sysargs, type = float, strElse = False):
    
    toReturn = val

    if not strElse:
        if cmd in sysargs:
            ind = sysargs.index(cmd) + 1
            val = type(sysargs[ind])
    else:
        if cmd in sysargs:
            ind = sysargs.index(cmd) + 1
            try:
                val = type(sysargs[ind])
            except:
                val = sysargs[ind]

    return val

def setRunParametersFromCommandLine(sysargs = []):

    geometryDesc = "NC"
    initPlan = "Plan20"
    numDists = 13
    idealPop = "mean"
    step = 0
    steps = 50000
    mul = 1
    seed = 912311
    gamma = 0
    popDivConstraint = 0.02
    compactnessWeight = 0
    maxSplitCounties = 13

    mul = readCmd("--mulSeed", mul, sysargs, int)
    step = readCmd("--step", step, sysargs, int)
    gamma = readCmd("--gamma", gamma, sysargs)
    initPlan = readCmd("--initPlan", initPlan, sysargs, str)
    numDists = readCmd("--numDists", numDists, sysargs, int)
    idealPop = readCmd("--idealPop", idealPop, sysargs, strElse = True)
    geometryDesc = readCmd("--geom", geometryDesc, sysargs, str)
    maxSplitCounties = readCmd("--maxCntySplt", maxSplitCounties, sysargs, int)
    popDivConstraint = readCmd("--popDivConstraint", popDivConstraint, sysargs)
    compactnessWeight = readCmd("--weightPP", compactnessWeight, sysargs)

    seed *= mul
    
    pathToData = os.path.join("..", "inputData", geometryDesc)
    state = {"graph" : districtingGraph.set(pathToData)}
    
    random.seed(seed)
    rng = random
    
    desc = "gamma" + str(gamma).replace(".", "p") + "_seed" + str(seed) + \
           "_popdiv" + str(popDivConstraint) + "_initPlan" + initPlan 

    # desc = "g0_initPlan" + initPlan + "maxCntySplt" + str(maxSplitCounties)
    if compactnessWeight > 0:
        desc += "_wc" + str(compactnessWeight)
    outDir = os.path.join("..","Output", geometryDesc, desc)
    
    args = {}
    args["rng"] = rng
    if compactnessWeight > 0:
        args["energy"] = {"compactWeight" : compactnessWeight}
    args["constraints"] = {"populationDeviation" : popDivConstraint}
    args["initPlan"] = initPlan
    args["parameters"] = {"gamma" : gamma, "idealPop" : idealPop, 
                          "districts" : numDists, "steps" : steps, 
                          "outDir" : outDir, 
                          "maxSplitCounties" : maxSplitCounties, 
                          "step" : step}
    # print(args)
    # print(seed)
    return state, args

