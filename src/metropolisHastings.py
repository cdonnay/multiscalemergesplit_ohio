import importlib
import math

import energy
import mergeSplit
import multiLayeredGraph
import stateExt 
import tree
import writer

importlib.reload(energy)
importlib.reload(mergeSplit)
importlib.reload(multiLayeredGraph)
importlib.reload(stateExt)
importlib.reload(tree)
importlib.reload(writer)

import multiLayeredGraph as mlg

exp = lambda x: math.exp(min(x, 700)) #>~700 overflows

################################################################################

def run(state, proposal, info):
    '''General MCMC districting. Energy function is (global) ENERGY_LOOKUP[i]
       if i is int, else energy function is i. Proposal function is (inputted) 
       proposal_f.'''
    rng = info["rng"]
    # print("info rng rs", info['rng'].uniform(0,1))
    # print("info rng run", rng.uniform(0,1))

    computeEnergy = energy.getEnergyFunction(info)
    state = stateExt.extendState(state, info, computeEnergy)
    initialStep = info["parameters"]["step"]
    finalStep = info["parameters"]["steps"]
    # print(initialStep)
    # exit()
    writer.setupOutputs(state, info)
    writer.recordState(initialStep, state, info)
    # writer.recordStateData(initialStep, state, info)

    for step in range(initialStep, finalStep):
        # print("step", step)
        # print("info rng run step", step, rng.uniform(0,1))
        (proposalData, prstntEdgeMap, p) = proposal(state, info)

        cutEdges = proposalData[4]
        # print(prstntEdgeMap)
        # if len(prstntEdgeMap)>0:
        #     print(state["layeredGraph"]["graphLevels"][0].nodes[list(prstntEdgeMap.keys())[0][0][0]]["County"], p)
        if cutEdges == None:
            continue
        # newdistricts["energy"] = computeEnergy(state, info, newdistricts)
        # p *= exp(state["energy"] - newdistricts["energy"])
        # print(proposalData[0])
        if rng.random() < p:
            print("acceptance", step)
            # for d in range(1,14):
            #     print("d", d, state["layeredGraph"]["nestedTrees"][d].keys())
            # print(proposalData[0])
            # print("johnston before", state["districts"]["n2d"]["johnston"])
            # print("7 before", state["districts"]["d2n"][7])
            state = stateExt.updateState(proposalData, prstntEdgeMap, state)
            # print("johnston after", state["districts"]["n2d"]["johnston"])
            # print("7 after", state["districts"]["d2n"][7])
            writer.recordState(step, state, info)
            # writer.recordStateData(step, state, info)
            # break
            # print("after acceptance")
            # for d in range(1,14):
            #     print("d", d, state["layeredGraph"]["nestedTrees"][d].keys())
            # dists = mlg.getDistricts(state, "mecklenburg", 1)
            # print("dists", dists)
            # print(proposalData[0])
            # exit_huh = False
            # for d in dists:
            #     if ("mecklenburg", 1, d) not in state["layeredGraph"]["topDistGraph"]:
            #         print(d, proposalData[0][d]["top"][0].nodes)
            #         exit_huh = True
            # if exit_huh:
            #     raise Exception("ERROR HERE AFTER")
        # nestedTrees = state["layeredGraph"]["nestedTrees"]
        # fineGraph = state["layeredGraph"]["graphLevels"][0]
        # for d in nestedTrees:
        #     for cntyLevel in nestedTrees[d]:
        #         if cntyLevel == "top":
        #             continue
        #         outEdges = nestedTrees[d][cntyLevel][2]
        #         outCntys = set([])
        #         for oE in outEdges:
        #             n1, n2 = list(oE)
        #             top1, top2 = n1[0], n2[0]
        #             if n1[1] == 0:
        #                 top1 = fineGraph.nodes[n1[0]]["County"]
        #             if n2[1] == 0:
        #                 top2 = fineGraph.nodes[n2[0]]["County"]
        #             outCnty = list(set([top1, top2]) - set([cntyLevel[0]]))[0]
        #             # print("cntyLevel/outCnty", d, cntyLevel, outCnty)
        #             if outCnty in outCntys:
        #                 print("PROBLEM!!!")
        #                 print(nestedTrees)
        #                 raise Exception(d, cntyLevel, step, outCnty)
        #             outCntys.add(outCnty)



    return state
