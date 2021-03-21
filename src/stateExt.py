import multiLayeredGraph as mlg
import networkx as nx
import tree

import importlib

importlib.reload(mlg)
importlib.reload(tree)

def setExtensionData(state, info):
    state["extensions"] = set([])
    if info["proposal"] == "mergeSplit":
        state["extensions"].add("spanningTrees")
        if info["parameters"]["gamma"] != 0:
            state["extensions"].add("spanningTreeCounts")

    if info["proposal"] == "multiScaleMergeSplitFine":
        state["extensions"].add("nestedTrees")
        if info["parameters"]["gamma"] != 0:
            state["extensions"].add("nestedTreeCounts")        
            state["extensions"].add("persistantEdgeCount")        
    return state

def extendState(state, info, computeE):
    state = setExtensionData(state, info)

    rng = info["rng"]

    ##### mergeSplit #####
    if "spanningTrees" in state["extensions"]:
        state["districtTrees"] = {}
        for di in state["districts"]:
            district = state["districts"][di]
            state["districtTrees"][di] = tree.wilson(district, rng)
    if "spanningTreeCounts" in state["extensions"]:
        state["spanningTreeCounts"] = {}
        for di in state["districts"]:
            district = state["districts"][di]
            state["spanningTreeCounts"][di] = tree.nspanning(district)

    ##### multiScaleMergeSplit #####
    if "nestedTrees" in state["extensions"]:
        nestedTrees = {}
        nestedHeirarchy = state["layeredGraph"]["hierarchy"]

        state["layeredGraph"]["nestedTrees"] = {}

        cntyGraph = state["layeredGraph"]["graphLevels"][1]
        fineGraph = state["layeredGraph"]["graphLevels"][0]
        for cnty in cntyGraph.nodes:
            nodes = cntyGraph.nodes[cnty]["nestedElements"]
            subGraph = fineGraph.subgraph(nodes)
            noSpanT = tree.nspanning(subGraph)
            cntyGraph.nodes[cnty]["noSpanT"] = noSpanT


        for dist in range(1, state["noDists"]+1):
            nestedTrees = {}
            distNodes = mlg.getTopLevelNodes(state, dist)
            topDistGraph = state["layeredGraph"]["topDistGraph"]
            topDistSubGraph = topDistGraph.subgraph(distNodes)
            # print("info rng cTop1 loop", dist, info['rng'].uniform(0,1))
            distTree = tree.wilson(topDistSubGraph, info["rng"])
            # print(distTree.nodes)
            # print("info rng cTop2 loop", info['rng'].uniform(0,1))

            noSpanT = tree.nspanning(topDistSubGraph)

            topLevel = max(state["layeredGraph"]["hierarchy"])
            splits = mlg.getSplitNodes(state, topLevel, dist = dist)
            splits = [(c, topLevel) for c in splits]

            nestedTrees["top"] = [distTree, topDistSubGraph, set([]), splits, 
                                  noSpanT] 
            # pop = sum([topDistSubGraph.nodes[n]["Population"] for n in topDistSubGraph.nodes])
            # print("dist pop", dist, pop, (pop-state['idealPop'])/state['idealPop'])
            # make recursive
            # state["extensions"]["nestedTrees"] = \
            #      buildNestedTrees(state["extensions"]["nestedTrees"], )
            for nodeLevel in sorted(splits):
                node, level = nodeLevel
                subGraph = mlg.buildSublevelDistGraph(state, node, level, dist)
                # print(subGraph.nodes)
                # print(subGraph.edges)
                # print("info rng ca loop", info['rng'].uniform(0,1))
                subTree = tree.wilson(subGraph, info["rng"])
                # print("info rng cb loop", info['rng'].uniform(0,1))

                splits = []
                # splits = mlg.getSplitNodes(state, level, dist, node)
                noSpanT = tree.nspanning(subGraph)
                nestedTrees[nodeLevel] = [subTree, subGraph, set([]), [], 
                                          noSpanT]

            for edge in sorted(distTree.edges):
                e1s = (edge[0][0], edge[0][1])
                e2s = (edge[1][0], edge[1][1])
                if e1s not in nestedTrees["top"][3] and \
                   e2s not in nestedTrees["top"][3]:
                   continue
                # print("info rng c3 loop", info['rng'].uniform(0,1))
                crossEdge = mlg.setCrossEdge(state, e1s, e2s, set([dist]), info)
                # print("info rng c4 loop", info['rng'].uniform(0,1))

                if e1s in nestedTrees["top"][3]:
                    nestedTrees[e1s][2].add(crossEdge)
                if e2s in nestedTrees["top"][3]:
                    nestedTrees[e2s][2].add(crossEdge)
            state["layeredGraph"]["nestedTrees"][dist] = nestedTrees

    state["energy"] = 0#computeE(state, info)

    return state

def updateState(proposalData, prstntEdgeMap, state):

    # for di in newdistricts["districts"]:
    #     district = newdistricts["districts"][di]
    #     state["districts"][di] = district

    # state["nodeToDistrict"] = newdistricts["nodeToDistrict"]
    
    if "spanningTrees" in state["extensions"]:
        for di in newdistricts["districtTrees"]:
            districtTree = newdistricts["districtTrees"][di]
            state["districtTrees"][di] = districtTree
    if "spanningTreeCounts" in state["extensions"]:
        for di in newdistricts["spanningTreeCounts"]:
            spanningTreeCount = newdistricts["spanningTreeCounts"][di]
            state["spanningTreeCounts"][di] = spanningTreeCount

    if "nestedTrees" in state["extensions"]:
        #update persistent edges
        # print("oldPE", state["layeredGraph"]["persistentEdges"])
        for oPE in prstntEdgeMap:
            nPE = prstntEdgeMap[oPE]
            state["layeredGraph"]["persistentEdges"] -= set([oPE])
            state["layeredGraph"]["persistentEdges"].add(nPE)
        # print("newPE", state["layeredGraph"]["persistentEdges"])
        
        #update d2n
        newDists, newDistAssignments, newNodeAssignments, newPerstEdg, cutEdges =\
                                                                    proposalData
        for d in newDistAssignments:
            # print("before", d, state["districts"]["d2n"][d])
            state["districts"]["d2n"][d] = newDistAssignments[d]
            # print("after", d, state["districts"]["d2n"][d])
        #update n2d
        # print("newNodeAssignments", newNodeAssignments)
        for n in newNodeAssignments:
            d = newNodeAssignments[n]
            # print("old", n, state["districts"]["n2d"][n], d)
            if isinstance(d, int):
                state["districts"]["n2d"][n] = d
            else:
                oldD = state["districts"]["n2d"][n]
                if not isinstance(oldD, int):
                    state["districts"]["n2d"][n][0] = d[0]
                    state["districts"]["n2d"][n][1].update(d[1])
                else:
                    state["districts"]["n2d"][n] = d
            # print("new", n, state["districts"]["n2d"][n])

        for d in newDists:
            state["layeredGraph"]["nestedTrees"][d] = newDists[d]

        #topDistGraph
        newTopDistGraph = nx.MultiGraph(state["layeredGraph"]["topDistGraph"])
        coarseGraph = state["layeredGraph"]["graphLevels"][1]
        fineGraph = state["layeredGraph"]["graphLevels"][0]
        dists = set(newDists.keys())
        nodesToRip = [n for n in newTopDistGraph.nodes if n[2] in dists]
        newTopDistGraph.remove_nodes_from(nodesToRip)
        for d in newDists:
            topDistSubGraph = newDists[d]["top"][1]
            if len(topDistSubGraph.nodes) == 0:
                newDists[d]["top"][1] = copy.deepcopy(newDists[d]["top"][0])
            #     pop = 0
            #     for topn in state["districts"]["d2n"][d][1]:
            #         if topn in state["districts"]["d2n"][d][0]:
            #             for botn in state["districts"]["d2n"][d][0][topn]:
            #                 pop += fineGraph.nodes[botn]["Population"]
            #     newDists[d]["top"][1][""]

            for n in topDistSubGraph.nodes:
                sgdists = mlg.getDistricts(state, n[0], 1)
                if isinstance(sgdists, int):
                    topDistSubGraph.nodes[n]["Population"] = \
                                           coarseGraph.nodes[n[0]]["Population"]
                else:
                    pop = 0
                    for sn in coarseGraph.nodes[n[0]]["nestedElements"]:
                        sgdist = mlg.getDistricts(state, sn, 0)
                        if sgdist == n[2]:
                            pop += fineGraph.nodes[sn]["Population"]
                    topDistSubGraph.nodes[n]["Population"] = pop

            newTopDistGraph = nx.compose(newTopDistGraph, topDistSubGraph)
            newTopDistGraph = nx.MultiGraph(newTopDistGraph)

            # currentCntys = set([n[0] for n in newTopDistGraph.nodes])

            for node in topDistSubGraph:
                #print("topDistSubGraph", node)
                cnty, level, dist = node
                
                for cntyNbr in coarseGraph.neighbors(cnty):
                    nbrDists = mlg.getDistricts(state, cntyNbr, 1)
                    if isinstance(nbrDists, int):
                        nbrDists = [nbrDists]
                    for nbrDist in nbrDists:
                        if nbrDist == d:
                            continue
                        if (cntyNbr, 1, nbrDist) not in newTopDistGraph.nodes:
                            continue
                        edgeCount = 0
                        
                        fineNbrs = coarseGraph.nodes[cnty]["downNbrs"][cntyNbr][1]
                        fineNbrs = fineNbrs.keys()
                        for fineNbr in fineNbrs:
                            fineNbrDist = mlg.getDistricts(state, fineNbr, level-1)
                            if fineNbrDist != nbrDist:
                                continue
                            for fineNbrNbr in fineGraph.neighbors(fineNbr):
                                topFineNbrNbr = fineGraph.nodes[fineNbrNbr]["County"]
                                if topFineNbrNbr != cnty:
                                    continue
                                fineDist = mlg.getDistricts(state, fineNbrNbr, level-1)
                                if fineDist == d:
                                    edgeCount += 1

                        for e in range(edgeCount):
                            newTopDistGraph.add_edge((cntyNbr, 1, nbrDist), node)
        # print(newDists.keys())
        # for node in newTopDistGraph.nodes:
        #     for nbr in newTopDistGraph.neighbors(node):
        #         if newTopDistGraph.number_of_edges(nbr, node) != coarseGraph.number_of_edges(nbr[0], node[0])\
        #             and node[0] < nbr[0]:

        #             print("newTopDistGraph", node, nbr, newTopDistGraph.number_of_edges(nbr, node),
        #                                      coarseGraph.number_of_edges(nbr[0], node[0]))

        nestedTrees = state["layeredGraph"]["nestedTrees"]
        for d in state["layeredGraph"]["nestedTrees"]:
            if len(nestedTrees[d]["top"][1].nodes) == 0:
                print("dist", d)
                print("modified dists", dists)
                print("newTopDistGraph", newTopDistGraph.nodes)
                print("topDistGraph", state["layeredGraph"]["topDistGraph"].nodes)
                print("nestedTree.nodes", d, nestedTrees[d]["top"][0].nodes)
                print("nestedGraph.nodes", d, nestedTrees[d]["top"][1].nodes)
                print("nestedTree", d, nestedTrees[d])
                print("nodesToRip", nodesToRip)
                for d in dists:
                    print("newDists", d, newDists[d])
                    print("newDists.top.Tree", newDists[d]["top"][0].nodes)
                    print("newDists.top.Graph", newDists[d]["top"][1].nodes)
                raise Exception("found empty subgraph", d)

        state["layeredGraph"]["topDistGraph"] = newTopDistGraph
    # state["energy"] = newdistricts["energy"]
    return state
