import numpy as np
import os


def makeStateDataHeaders(state, info):
    stateDataHeaders = ["step"]

    noDistricts = info["parameters"]["districts"]
    if "spanningTreeCounts" in state["extensions"]:
        for di in range(noDistricts):
            stateDataHeaders += ["spanningTreeCounts_dist" + str(di)]

    return stateDataHeaders

def makeStateDataArray(step, state, info):
    stateDataArray = [step]

    noDistricts = info["parameters"]["districts"]
    if "spanningTreeCounts" in state["extensions"]:
        for di in range(noDistricts):
            noSpanningTrees = np.round(np.exp(state["spanningTreeCounts"][di]))
            stateDataArray += [int(noSpanningTrees)]

    return stateDataArray

def setupOutputs(state, info, delim = '\t'):
    try:
        outDir = info["parameters"]["outDir"]
    except:
        return

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    outDirSamples = os.path.join(outDir, "Samples")
    if not os.path.exists(outDirSamples):
        os.makedirs(outDirSamples)

    stateDataHeaders = makeStateDataHeaders(state, info)
    metaDataPath = os.path.join(outDir, "metaData.txt")
    with open(metaDataPath, "w") as mdf:
        mdf.write(delim.join([str(sdh) for sdh in stateDataHeaders]) + "\n")
    
def recordNodes(outFile, nodeToDists, level, delim):
    for n in nodeToDists:
        if isinstance(nodeToDists[n], int):
            toWrite = [str(n), str(level), str(nodeToDists[n])]
            outFile.write(delim.join(toWrite)+ "\n")
        else:
            subNodeToDists = nodeToDists[n][1]
            recordNodes(outFile, subNodeToDists, level-1, delim)

def recordState(step, state, info, delim = '\t'):
    try:
        outDir = info["parameters"]["outDir"]
    except:
        return
    outDirSamples = os.path.join(outDir, "Samples")
    outPath = os.path.join(outDirSamples, str(step) + ".txt")
    outFile = open(outPath, "w")

    topLevel = max(state["layeredGraph"]["hierarchy"])
    recordNodes(outFile, state["districts"]["n2d"], topLevel, delim)

    outFile.close()

def recordStateData(step, state, info, delim = '\t'):
    try:
        outDir = info["parameters"]["outDir"]
    except:
        return

    stateDataArray = makeStateDataArray(step, state, info)
    metaDataPath = os.path.join(outDir, "metaData.txt")
    with open(metaDataPath, "a") as mdf:
        mdf.write(delim.join([str(sdh) for sdh in stateDataArray]) + "\n")

