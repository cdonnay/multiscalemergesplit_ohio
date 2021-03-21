import copy
import math
import networkx as nx

import multiLayeredGraph as mlg
import tree

import importlib

importlib.reload(mlg)
importlib.reload(tree)

exp = lambda x: math.exp(min(x, 700)) #>~700 overflows

################################################################################

def parsePersistentEdgeDists(state, persistentEdge):
    e, cnt, levels = persistentEdge
    l1, l2 = levels
    g1 = state["layeredGraph"]["graphLevels"][l1]
    d1 = mlg.getDistricts(state, e[0], l1)
    g2 = state["layeredGraph"]["graphLevels"][l2]
    d2 = mlg.getDistricts(state, e[1], l2)
    return d1, d2

################################################################################

def getMergedTree(state, info, d1, d2):
    districts, graph = state["districts"], state["graph"]
    mergedNodes = set(districts[d1]).union(set(districts[d2]))
    mergedGraph = graph.subgraph(mergedNodes)
    mergedTree = tree.wilson(mergedGraph, info["rng"])
    mergedPop = sum([graph.nodes[n]["Population"] for n in mergedTree])
    return mergedTree, mergedPop

################################################################################

def proposeCut(mergedTree, mergedPop, mergedGraph, state, info, dists, 
               persistentEdge, level):

    cutEdges, edgeWeights, root = tree.edgeCuts(mergedTree, mergedPop, 
                                                mergedGraph, state, 
                                                retRoot = True)

    topLevel = max(state["layeredGraph"]["hierarchy"])
    topDesc = state["layeredGraph"]["hierarchy"][level]
    cutNodesInfo = mlg.findCutNodes(edgeWeights, root, mergedPop, 
                                    mergedGraph, state, level, dists)
    nestedTrees = state["layeredGraph"]["nestedTrees"]

    fineGraph = state["layeredGraph"]["graphLevels"][0]
    coarseGraph = state["layeredGraph"]["graphLevels"][1]
    cnty1 = fineGraph.nodes[persistentEdge[0][0]][topDesc]
    cnty2 = fineGraph.nodes[persistentEdge[0][1]][topDesc]
    mergedCnty = set([])
    if cnty1 == cnty2:
        mergedCnty.add((cnty1, 1))
    
    cutNodeToTree = {}
    cutNodeSet = set([cn[0] for cn in cutNodesInfo])

    for cnty in mergedTree.nodes:
        cntyDists = mlg.getDistricts(state, cnty[0], topLevel)
        if isinstance(cntyDists, int):
            continue
        cntyDists = set(cntyDists)
        if len(cntyDists-dists)>0:
            cutNodeSet.add(cnty)
    
    cutEdgesTop = set([])
    outEdges = {}
    topOutEdgeToFineOutEdge = {}
    for edge in cutEdges:
        n1, n2 = list(edge)
        n1In = n1 in cutNodeSet
        n2In = n2 in cutNodeSet
        if n1In or n2In:
            edge = mlg.setCrossEdge(state, n1, n2, dists, info, 
                                    cut12 = {1: n1In, 2: n2In})
        if n1 in outEdges:
            outEdges[n1].add(edge)
        else:
            outEdges[n1] = set([edge])
        if n2 in outEdges:
            outEdges[n2].add(edge)
        else:
            outEdges[n2] = set([edge])
        sn1, sn2 = list(edge)
        cutEdgesTop.add(frozenset([sn1[0], sn2[0]]))
        topOutEdgeToFineOutEdge[frozenset([n1, n2])] = edge
    
    # print("cutEdgesTop", cutEdgesTop)
    # print("outEdges1", outEdges)

    # print(cutEdges)
    cutEdges = set([])
    cutEdges.update(cutEdgesTop)
    
    for cutNodeInfo in cutNodesInfo:
        cutNode, nbrSplits, nbrToPopRep = cutNodeInfo
        # print("cutNode", cutNode)
        # print("nbrSplits", nbrSplits)
        coarseGraph = state["layeredGraph"]["graphLevels"][cutNode[1]]
        fineGraph = state["layeredGraph"]["graphLevels"][cutNode[1]-1]
        subNodes = coarseGraph.nodes[cutNode[0]]["nestedElements"]
        subGraph = fineGraph.subgraph(subNodes)
        subTree = tree.wilson(subGraph, info["rng"])

        for n in subTree.nodes:
            subTree.nodes[n]["Population"] = subGraph.nodes[n]["Population"]

        cutNodeToTree[cutNode] = subTree

        addedNodes = set([])
        addedEdges = set([])
        for nbr in nbrToPopRep:
            edge = frozenset([nbr, cutNode])
            if edge in topOutEdgeToFineOutEdge:
                edge = topOutEdgeToFineOutEdge[edge]
                n1, n2 = list(edge)
                # print("edge", edge)
                if n1[1] != topLevel:
                    g = state["layeredGraph"]["graphLevels"][n1[1]]
                    topName1 = g.nodes[n1[0]][topDesc]
                else:
                    topName1 = n1[0]
                if n2[1] != topLevel:
                    g = state["layeredGraph"]["graphLevels"][n2[1]]
                    topName2 = g.nodes[n2[0]][topDesc]
                else:
                    topName2 = n2[0]

                if topName1 == cutNode[0]:
                    outNode = (topName2, topLevel)
                    inNode = n1[0]
                else:
                    outNode = (topName1, topLevel)
                    inNode = n2[0]
                addedNodes.add(outNode)
                addedEdges.add(frozenset([inNode, outNode]))
                subTree.add_edge(outNode, inNode)
                outEdgePop = nbrToPopRep[outNode]
                subTree.nodes[outNode]["Population"] = outEdgePop
            else:
                cut2 = False
                # print("nbr", nbr)
                if nbr in cutNodeSet:
                    cut2 = True
                edge = mlg.setCrossEdge(state, cutNode, nbr, dists, info, 
                                        cut12 = {1: True, 2: cut2})
                n1, n2 = list(edge)
                top1, top2 = n1[0], n2[0]
                if n1[1] == 0:
                    top1 = fineGraph.nodes[n1[0]][topDesc]
                if n2[1] == 0:
                    top2 = fineGraph.nodes[n2[0]][topDesc]
                if (top1,1) == nbr:
                    node = n2[0]
                else:
                    node = n1[0]
                addedNodes.add(nbr)
                addedEdges.add(frozenset([node, nbr]))
                subTree.add_edge(node, nbr)
                outEdgePop = nbrToPopRep[nbr]
                subTree.nodes[nbr]["Population"] = outEdgePop
                # print("adding out edge", edge)
                if cutNode in outEdges:
                    outEdges[cutNode].add(edge)
                else:
                    outEdges[cutNode] = set([edge])


        subNodeCutInfo = tree.edgeCuts(subTree, mergedPop, subTree, state, 
                                       retRoot = True)
        cutEdges.update(subNodeCutInfo[0] - addedEdges)
        # print(cutNodeInfo, subNodeCutInfo[0] - addedEdges)
        subTree.remove_nodes_from(addedNodes)
    
    # print("cutEdges", cutEdges)
    if len(cutEdges) == 0:
        return {}, {}, {}, (), set([])


    cutEdge = info['rng'].choice(sorted(list(cutEdges)))

########
    # print("DEBUGGING HERE -- forcing cnty to cnty edge")
    # cutEdge = info['rng'].choice(sorted(list(cutEdgesTop)))
    # print("DEBUGGING HERE -- forcing inner cnty edge")
    # cutEdge = info['rng'].choice(sorted(list(cutEdges - cutEdgesTop)))
########
    # print("cutNodesInfo", [c[0] for c in cutNodesInfo])
    # print(mergedTree.edges)
    # print("cutEdgesTop", cutEdgesTop)
    # print("cutEdges", cutEdges)

    print_huh = False
    # if 8 in dists:
    #     print_huh = True
    # print("cutEdge", cutEdge)
    # print("outEdges", outEdges)

    newDists = {}
    newDistAssignment = {d : {1 : set([]), 0 : {}} for d in dists}
    newNodeAssignment = {}

    fineGraph = state["layeredGraph"]["graphLevels"][0]
    n1, n2 = list(cutEdge)

    # cntysInCut = set([fineGraph.nodes[n1][topDesc], fineGraph.nodes[n2][topDesc]])
    # if "rowan" in cntysInCut:
    #     print_huh = True    
    #     print("cnties in cut", cntysInCut, cutEdge in cutEdgesTop)
    #     print("mergedTree", mergedTree.edges)
    #     print("outEdges", ("rowan", 1), outEdges[("rowan", 1)])
    #     print("rowan tree", cutNodeToTree[("rowan", 1)].edges)
    ### if cut cnty the do this
    if cutEdge in cutEdgesTop:
        # print("on cnty")

        nc1 = (fineGraph.nodes[n1][topDesc], 1)
        nc2 = (fineGraph.nodes[n2][topDesc], 1)
        mergedTree.remove_edge(nc1, nc2)
        newTopTrees = [mergedTree.subgraph(n) 
                       for n in nx.connected_components(mergedTree)]
        
        # assign trees and subgraphs
        nestedTrees = state["layeredGraph"]["nestedTrees"]
        for ii, di in enumerate(dists):
            # print(di, newTopTrees[ii].nodes)
            nodeMap = {n : n + (di,) for n in newTopTrees[ii].nodes}
            newTopTrees[ii] = nx.relabel_nodes(newTopTrees[ii], nodeMap)
            newDists[di] = {"top" : [newTopTrees[ii], "", set([]), [], 0]}
            for di2 in dists:
                oldDistNodes = set([])#set(nestedTrees[di2].keys()) - mergedCnty
                for node in nestedTrees[di2].keys():
                    if node == "top":
                        continue
                    keyDists = mlg.getDistricts(state, node[0], topLevel)
                    if not isinstance(keyDists, int) and \
                       set(dists) != set(keyDists):
                       oldDistNodes.add(node)
                # oldDistNodes.remove(mergedCnty)
                newDistNodes = set(nodeMap.keys())
                distNodes = oldDistNodes.intersection(newDistNodes)
                if len(distNodes) == 0:
                    continue
                for dn in distNodes:
                    # print(di, dn, len(nestedTrees[di2][dn]))
                    newDists[di][dn] = copy.deepcopy(nestedTrees[di2][dn])
                    # newDists[di][dn] = nestedTrees[di2][dn]
                    newDists[di][dn][2] = set([])
                    newDists[di]["top"][3].append(dn)

                # print("d's", di, di2)
                # print("oldDistNodes", oldDistNodes)
                # print("newDistNodes", newDistNodes)
                # print("distNodes", distNodes)
                # print("distNodes", mergedCnty)
                # raise Exception("here")

            for cutNodeInfo in cutNodesInfo:
                cutNode = cutNodeInfo[0]
                if cutNode not in nodeMap.keys():
                    continue
                # print("cutNode add explicit", cutNode, di)
                subGraphNodes = coarseGraph.nodes[cutNode[0]]["nestedElements"]
                subGraph = fineGraph.subgraph(subGraphNodes)
                noSpanT = coarseGraph.nodes[cutNode[0]]["noSpanT"]
                newDists[di][cutNode] = [cutNodeToTree[cutNode], subGraph, 
                                         set([]), [], noSpanT]
                newDists[di]["top"][3].append(cutNode)
        
        # for d in newDists:
        #     print("tree", d, newDists[d]["top"][0].nodes)

        #update district assignments in newDistAssignment
        for diOld in newDists:
            oldAssignment = state["districts"]["d2n"][diOld]
            
            cntys = oldAssignment[1]
            for cnty in cntys:
                node = (cnty, 1)
                for diNew in newDists:
                    if node + (diNew,) in newDists[diNew]["top"][0].nodes:
                        # print("adding", cnty, "from", diOld, "to", diNew)
                        newDistAssignment[diNew][1].add(cnty)
                        newNodeAssignment[cnty] = diNew
        

            splitCntys = oldAssignment[0]
            for splitCnty in splitCntys:
                node = (splitCnty, 1)
                oldDists = state["districts"]["n2d"][splitCnty][0]
                # print("dealing with", splitCnty, "from old dist", oldDists)
                if isinstance(oldDists, int):
                    oldDists = [oldDists]
                oldDists = set(oldDists)

                # print(dists, oldDists)

                if oldDists != dists:
                    # print("inside")
                    oldDist = list(dists.intersection(oldDists))[0]
                    for diNew in newDists:
                        # print("is in???", node, newDists[diNew]["top"][0].nodes, node in newDists[diNew]["top"][0].nodes)
                        if node + (diNew,) in newDists[diNew]["top"][0].nodes:
                            # print(splitCnty, "was in", oldDist, "and is now in", diNew)
                            newDist = diNew
                    pctsInOldDist = oldAssignment[0][splitCnty]
                    newDistAssignment[newDist][0][splitCnty] = pctsInOldDist
                    # if oldDist == newDist:
                    #     continue
                    otherDist = list(oldDists - dists)[0]
                    newNodeAssignment[splitCnty] = [[otherDist, newDist], {}]
                    oldPctAssignment = state["districts"]["n2d"][splitCnty][1]
                    for pct in pctsInOldDist:
                        newNodeAssignment[splitCnty][1][pct] = newDist


        # build top dist graphs
        for di in newDists:
            nodes = [n[0] for n in newDists[di]["top"][0].nodes]
            # print("di nodes", di, nodes)
            subTopDistGraph = nx.MultiGraph(coarseGraph.subgraph(nodes))
            # accountedPairs = set([])
            for node in nodes:
                cntyDists = mlg.getDistricts(state, node, level, 
                                             updateDist = newNodeAssignment)
                if isinstance(cntyDists, int):
                    continue
                # print("cntyDists", node, cntyDists)
                subTopDistGraph.remove_node(node)

                for nbr in coarseGraph.neighbors(node):
                    if nbr not in nodes:
                        continue
                    edgeCount = 0
                    nbrDists = mlg.getDistricts(state, node, level, 
                                                updateDist = newNodeAssignment)
                    if isinstance(nbrDists, int):
                        nbrDists = [nbrDists]
                    if di not in nbrDists:
                        continue
                    fineNbrs = coarseGraph.nodes[node]["downNbrs"][nbr][1]
                    fineNbrs = fineNbrs.keys()
                    for fineNbr in fineNbrs:
                        nbrDist = mlg.getDistricts(state, fineNbr, level-1, 
                                                 updateDist = newNodeAssignment)
                        if nbrDist != di:
                            continue
                        for fineNbrNbr in fineGraph.neighbors(fineNbr):
                            topFineNbrNbr = fineGraph.nodes[fineNbrNbr][topDesc]
                            if topFineNbrNbr != node:
                                continue
                            fineDist = mlg.getDistricts(state, fineNbrNbr, level-1, 
                                                 updateDist = newNodeAssignment)
                            if fineDist == di:
                                edgeCount += 1

                    for e in range(edgeCount):
                        subTopDistGraph.add_edge(nbr, node)


                        # print("fineNbrs", node, nbr, fineNbr, nbrDist)


                # pop = 0
                # if node in newDistAssignment[di][0]:
                #     for subNode in newDistAssignment[di][0][node]:
                #         fineGraph.nodes[subNode]["Population"]
                # if 
            if len(subTopDistGraph.nodes)==0:
                subTopDistGraph = nx.MultiGraph(coarseGraph.subgraph(nodes))
            nodeMap = {n: (n, 1, di) for n in nodes}
            subTopDistGraph = nx.relabel_nodes(subTopDistGraph, nodeMap)
            newDists[di]["top"][1] = subTopDistGraph

            if len(subTopDistGraph.nodes)==0:
                print("problem here", di, nodes, subTopDistGraph, newDists[di]["top"][0].nodes)
                raise Exception("Problem here")

        #update explicit out edges
        assignedEdges = set([])
        for nodeWithOutEdge in outEdges:
            for di in newDists:
                if nodeWithOutEdge in newDists[di]:
                    for existingOutEdge in outEdges[nodeWithOutEdge]:
                        cen1, cen2 = list(cutEdge)
                        cEwlevel = frozenset([(cen1,0), (cen2,0)])
                        if existingOutEdge == cEwlevel:
                            # print("found cut edge; skipping.", existingOutEdge, nodeWithOutEdge)
                            continue

                        newDists[di][nodeWithOutEdge][2].add(existingOutEdge)
                        # print("existingOutEdge", existingOutEdge, di)
                        n1, n2 = list(existingOutEdge)
                        cnty1, cnty2 = n1[0], n2[0]
                        if n1[1] == 0:
                            cnty1 = fineGraph.nodes[n1[0]][topDesc]
                        if n2[1] == 0:
                            cnty2 = fineGraph.nodes[n2[0]][topDesc]
                        othNode = set([(cnty1, 1), (cnty2, 1)]) 
                        othNode -= set([nodeWithOutEdge])
                        othNode = list(othNode)[0]
                        # print("othNode", othNode, othNode in newDists[di])
                        if othNode in newDists[di]:
                            newDists[di][othNode][2].add(existingOutEdge)


        for di in newDists:
            # print("district", di, newDists[di])
            for cntyWExpTree in newDists[di]:
                if cntyWExpTree == "top":
                    continue
                existingOutEdges = newDists[di][cntyWExpTree][2]
                topExistingEdgeNodes = set([])
                for eOEdge in existingOutEdges:
                    n1, n2 = list(eOEdge)
                    if n1[1] == 0:
                        n1 = (fineGraph.nodes[n1[0]][topDesc], 1)
                    if n2[1] == 0:
                        n2 = (fineGraph.nodes[n2[0]][topDesc], 1)
                    topExistingEdgeNodes.add(n1)
                    topExistingEdgeNodes.add(n2)
                # print("cntyWExpTree", cntyWExpTree, topExistingEdgeNodes)
                for nbr in newDists[di]["top"][0].neighbors(cntyWExpTree + (di,)):
                    nbr = nbr[:2]
                    if nbr in topExistingEdgeNodes:
                        continue
                    # print("edge to create", cntyWExpTree, nbr)
                    cut2 = False
                    if nbr in newDists[di]:
                        cut2 = True
                    edge = mlg.setCrossEdge(state, cntyWExpTree, nbr, dists, 
                                            info, cut12 = {1: True, 2: cut2}, 
                                            updateDist = newNodeAssignment)
                    newDists[di][cntyWExpTree][2].add(edge)
                    n1, n2 = list(edge)
                    cnty1, cnty2 = n1[0], n2[0]
                    if n1[1] == 0:
                        cnty1 = fineGraph.nodes[n1[0]][topDesc]
                    if n2[1] == 0:
                        cnty2 = fineGraph.nodes[n2[0]][topDesc]
                    othNode = set([(cnty1, 1), (cnty2, 1)]) 
                    othNode -= set([cntyWExpTree])
                    othNode = list(othNode)[0]
                    # print("othNode", othNode, othNode in newDists[di])
                    if othNode in newDists[di]:
                        newDists[di][othNode][2].add(edge)

        # update persistent edge possibilities
        d1, d2 = list(dists)
        noPossibleEdges = 0
        # print("handling persistentEdge")
        topGraph = state["layeredGraph"]["topDistGraph"]
        for node in newDists[d1]["top"][0].nodes:
            node = node[:2]
            for di in dists:
                tgNode = node + (di,)
                if tgNode not in topGraph.nodes:
                    continue
                for nbr in topGraph.neighbors(tgNode):
                    if nbr[0] not in newNodeAssignment:
                        continue
                    newNbrDists = newNodeAssignment[nbr[0]]
                    if isinstance(newNbrDists, int):
                        newNbrDists = [newNbrDists]
                    else:
                        newNbrDists = newNbrDists[0]
                    if d2 in newNbrDists:
                        edgeCount = topGraph.number_of_edges(tgNode, nbr)
                        noPossibleEdges += edgeCount
                        # print(node, nbr, di, newNbrDists, edgeCount)
                    # print(node, nbr, newDist)


        #build a tree on the subgraph of the county/node
        #connect the tree to the nbrs in nbrToPopRep
        #find edges to cut on
        #need to save the tree, with elements leaving the county
    else:
        # print("in else")
        #cutting within a node now
        cntyLevel = (fineGraph.nodes[n1][topDesc], 1)
        # if print_huh:
        # print("outEdges", cntyLevel, outEdges[cntyLevel])
        #     print(mergedTree.edges)
        #     print("cutable nodes", [c[0] for c in cutNodesInfo])
        # print(cutEdge)
        # print(cntyLevel)

        cntyLevelNbrs = set(mergedTree.neighbors(cntyLevel))
        mergedTree.remove_node(cntyLevel)
        lowTree = cutNodeToTree[cntyLevel]
        cn1, cn2 = list(cutEdge)
        lowTree.remove_edge(cn1, cn2)
        lowNodesToRemove = set([])
        for oE in outEdges[cntyLevel]:
            oE1, oE2 = list(oE)
            # print("oE", oE)
            top1, top2 = oE1[0], oE2[0]
            if oE1[1] == 0:
                top1 = fineGraph.nodes[oE1[0]][topDesc]
            if oE2[1] == 0:
                top2 = fineGraph.nodes[oE2[0]][topDesc]
            #
            if top1 != cntyLevel[0]:
                oE1 = (top1, 1)
                lowNodesToRemove.add(oE1)
            else:
                oE1 = oE1[0]
            if top2 != cntyLevel[0]:
                oE2 = (top2, 1)
                lowNodesToRemove.add(oE2)
            else:
                oE2 = oE2[0]
            # if print_huh:
            #     print("adding", oE1, oE2)
            lowTree.add_edge(oE1, oE2)

        cutLowTrees = {ii : lowTree.subgraph(n) 
                       for ii, n in enumerate(nx.connected_components(lowTree))}
        cutTopTreesList = [mergedTree.subgraph(n) 
                           for n in nx.connected_components(mergedTree)]
        cutTopTrees = [nx.Graph() for ii in range(len(cutLowTrees))]
        
        for ii in range(len(cutTopTrees)):
            cutTopTrees[ii].add_node(cntyLevel)

        # if print_huh:
        #     print("cntyLevel", cntyLevel)

        # print("lowNodesToRemove", lowNodesToRemove)

        for brdyNode in lowNodesToRemove:
            for ii in range(len(cutLowTrees)):
                if brdyNode not in cutLowTrees[ii].nodes:
                    continue
                for jj in range(len(cutTopTreesList)):
                    if brdyNode in cutTopTreesList[jj].nodes:
                        cutTopTrees[ii] = nx.compose(cutTopTrees[ii], 
                                                     cutTopTreesList[jj])

        #remove nodes added to each cut of low tree
        #add edges to back in from cut node
        for ii in range(len(cutLowTrees)):
            lowTree = nx.Graph(cutLowTrees[ii])
            lowTree.remove_nodes_from(lowNodesToRemove)
            cutLowTrees[ii] = lowTree

            backEdges = cntyLevelNbrs.intersection(set(cutTopTrees[ii].nodes))
            backEdges-= set([cntyLevel])
            for bEN in backEdges:
                cutTopTrees[ii].add_edge(cntyLevel, bEN)

        # for ii in range(len(cutTopTrees)):
        #     print("cutTopTrees.nodes", ii, cutTopTrees[ii].nodes)
        #     print("cutTopTrees.edges", ii, cutTopTrees[ii].edges)
        #     print()

        # assign trees and subgraphs
        nestedTrees = state["layeredGraph"]["nestedTrees"]
        for ii, di in enumerate(dists):
            # print(di, newTopTrees[ii].nodes)
            #need subgraph and then spanning tree count
            nodeMap = {n : n + (di,) for n in cutTopTrees[ii].nodes}
            cutTopTrees[ii] = nx.relabel_nodes(cutTopTrees[ii], nodeMap)
            newDists[di] = {"top" : [cutTopTrees[ii], "", set([]), [], 0]}
            for di2 in dists:
                oldDistNodes = set([])#set(nestedTrees[di2].keys()) - mergedCnty
                for node in nestedTrees[di2].keys():
                    if node == "top":
                        continue
                    keyDists = mlg.getDistricts(state, node[0], topLevel)
                    if not isinstance(keyDists, int) and \
                       set(dists) != set(keyDists):
                       oldDistNodes.add(node)
                # oldDistNodes.remove(mergedCnty)
                newDistNodes = set(nodeMap.keys()) - set([cntyLevel])
                distNodes = oldDistNodes.intersection(newDistNodes)
                if len(distNodes) == 0:
                    continue
                for dn in distNodes:
                    # print(di, dn, len(nestedTrees[di2][dn]))
                    newDists[di][dn] = copy.deepcopy(nestedTrees[di2][dn])
                    # newDists[di][dn] = nestedTrees[di2][dn]
                    newDists[di][dn][2] = set([])
                    newDists[di]["top"][3].append(dn)

                # print("d's", di, di2)
                # print("oldDistNodes", oldDistNodes)
                # print("newDistNodes", newDistNodes)
                # print("distNodes", distNodes)
                # print("distNodes", mergedCnty)
                # raise Exception("here")

            for cutNodeInfo in cutNodesInfo:
                cutNode = cutNodeInfo[0]
                if cutNode == cntyLevel:
                    continue
                if cutNode not in nodeMap.keys():
                    continue
                # print("cutNode add explicit", cutNode, di)
                subGraphNodes = coarseGraph.nodes[cutNode[0]]["nestedElements"]
                subGraph = fineGraph.subgraph(subGraphNodes)
                noSpanT = coarseGraph.nodes[cutNode[0]]["noSpanT"]
                newDists[di][cutNode] = [cutNodeToTree[cutNode], subGraph, 
                                         set([]), [], noSpanT]
                newDists[di]["top"][3].append(cutNode)

            subGraph = fineGraph.subgraph(cutLowTrees[ii].nodes)
            newDists[di][cntyLevel] = [cutLowTrees[ii], subGraph, set([]), 
                                       [], 0]
            newDists[di]["top"][3].append(cntyLevel)
        
        # for d in newDists:
        #     for key in newDists[d]:
        #         print("newDists", d, key, newDists[d][key])
        #         print("nodes", newDists[d][key][0].nodes)
        #         print("edges", newDists[d][key][0].edges)
        #     # for key in newDists[d]:
        #     #     print(d, key)
        #     #     print("newDists", d, key, newDists[d][key][1].nodes)
        # print("here ------ ")
        # state, newDists, 
        #update district assignments in newDistAssignment
        for diOld in newDists:

            oldAssignment = state["districts"]["d2n"][diOld]
            
            cntys = oldAssignment[1]
            for cnty in cntys:
                node = (cnty, 1)
                for diNew in newDists:
                    if node + (diNew,) in newDists[diNew]["top"][0].nodes:
                        # print("adding", cnty, "from", diOld, "to", diNew)
                        newDistAssignment[diNew][1].add(cnty)
                        # print("cnty/cntylevel[0]", cnty, cntyLevel[0])
                        if cnty not in newNodeAssignment and \
                           cnty != cntyLevel[0]:
                            newNodeAssignment[cnty] = diNew

            splitCntys = oldAssignment[0]
            for splitCnty in splitCntys:
                node = (splitCnty, 1)
                oldDists = state["districts"]["n2d"][splitCnty][0]
                # print("dealing with", splitCnty, "from old dist", oldDists)
                if isinstance(oldDists, int):
                    oldDists = [oldDists]
                oldDists = set(oldDists)

                # print(dists, oldDists)

                if oldDists != dists:
                    # print("inside")
                    oldDist = list(dists.intersection(oldDists))[0]
                    for diNew in newDists:
                        # print("is in???", node, newDists[diNew]["top"][0].nodes, node in newDists[diNew]["top"][0].nodes)
                        if node + (diNew,) in newDists[diNew]["top"][0].nodes:
                            # print(splitCnty, "was in", oldDist, "and is now in", diNew)
                            newDist = diNew
                    pctsInOldDist = oldAssignment[0][splitCnty]
                    newDistAssignment[newDist][0][splitCnty] = pctsInOldDist
                    # if oldDist == newDist:
                    #     continue
                    otherDist = list(oldDists - dists)[0]
                    newNodeAssignment[splitCnty] = [[otherDist, newDist], {}]
                    oldPctAssignment = state["districts"]["n2d"][splitCnty][1]
                    for pct in pctsInOldDist:
                        newNodeAssignment[splitCnty][1][pct] = newDist

            diNew = diOld
            subCntyNodes = set(newDists[diNew][cntyLevel][0].nodes)
            # print("assigning diNew", diNew, cntyLevel)
            newDistAssignment[diNew][0][cntyLevel[0]] = subCntyNodes
            if cntyLevel[0] not in newNodeAssignment:
                d1, d2 = list(dists)
                newNodeAssignment[cntyLevel[0]] = [[d1,d2], {}]
            
            # print("newNodeAssignment", newNodeAssignment)
            for pct in subCntyNodes:
                # print("pct in subCntyNodes", pct, cntyLevel)
                # print()
                newNodeAssignment[cntyLevel[0]][1][pct] = diNew

        # print("newDistAssignment", newDistAssignment)
        # print("newNodeAssignment", newNodeAssignment)
        # print

        #create subgraphs
        for di in newDists:
            nodes = [n[0] for n in newDists[di]["top"][0].nodes]
            # print("di nodes", di, nodes)
            subTopDistGraph = nx.MultiGraph(coarseGraph.subgraph(nodes))
            # accountedPairs = set([])

            for node in nodes:
                if len(nodes) == 1:
                    break
                cntyDists = mlg.getDistricts(state, node, level, 
                                             updateDist = newNodeAssignment)
                if isinstance(cntyDists, int):
                    continue
                # print("cntyDists", node, cntyDists)
                subTopDistGraph.remove_node(node)

                for nbr in coarseGraph.neighbors(node):
                    if nbr not in nodes:
                        continue
                    edgeCount = 0
                    nbrDists = mlg.getDistricts(state, node, level, 
                                                updateDist = newNodeAssignment)
                    if isinstance(nbrDists, int):
                        nbrDists = [nbrDists]
                    if di not in nbrDists:
                        continue
                    fineNbrs = coarseGraph.nodes[node]["downNbrs"][nbr][1]
                    fineNbrs = fineNbrs.keys()
                    for fineNbr in fineNbrs:
                        nbrDist = mlg.getDistricts(state, fineNbr, level-1, 
                                                 updateDist = newNodeAssignment)
                        if nbrDist != di:
                            continue
                        for fineNbrNbr in fineGraph.neighbors(fineNbr):
                            topFineNbrNbr = fineGraph.nodes[fineNbrNbr][topDesc]
                            if topFineNbrNbr != node:
                                continue
                            fineDist = mlg.getDistricts(state, fineNbrNbr, level-1, 
                                                 updateDist = newNodeAssignment)
                            if fineDist == di:
                                edgeCount += 1

                    for e in range(edgeCount):
                        subTopDistGraph.add_edge(nbr, node)


                        # print("fineNbrs", node, nbr, fineNbr, nbrDist)


                # pop = 0
                # if node in newDistAssignment[di][0]:
                #     for subNode in newDistAssignment[di][0][node]:
                #         fineGraph.nodes[subNode]["Population"]
                # if 

            if len(subTopDistGraph.nodes)==0:
                subTopDistGraph = nx.MultiGraph(coarseGraph.subgraph(nodes))
            nodeMap = {n: (n, 1, di) for n in nodes}
            subTopDistGraph = nx.relabel_nodes(subTopDistGraph, nodeMap)
            newDists[di]["top"][1] = subTopDistGraph

            if len(subTopDistGraph.nodes)==0:
                print("problem here", di, nodes, subTopDistGraph, newDists[di]["top"][0].nodes)
                raise Exception("Problem here")


            # print(cntyLevel)
            # for node in subTopDistGraph.nodes:
            #     for nbr in subTopDistGraph.neighbors(node):
            #         print("subgraph props", di, node, nbr, subTopDistGraph.number_of_edges(nbr, node),
            #                         coarseGraph.number_of_edges(nbr[0], node[0]))

        # for d in newDists:
        #     print("newDists[d].keys()", d, newDists[d].keys())
        #     if ("iredell", 1) in newDists[d]:
        #         print("iredell outedges before", d, newDists[d][("iredell", 1)][2])
        #         print("iredell outedges before", d, newDists[d][("iredell", 1)][0].nodes)
        #update explicit out edges
        assignedEdges = set([])
        for nodeWithOutEdge in outEdges:
            for di in newDists:
                if nodeWithOutEdge not in newDists[di]:
                    continue
                if nodeWithOutEdge == cntyLevel:
                    continue
                for existingOutEdge in outEdges[nodeWithOutEdge]:
                    cen1, cen2 = list(cutEdge)
                    cEwlevel = frozenset([(cen1,0), (cen2,0)])
                    if existingOutEdge == cEwlevel:
                        # print("found cut edge; skipping.", existingOutEdge, nodeWithOutEdge)
                        continue
                    newDists[di][nodeWithOutEdge][2].add(existingOutEdge)
                    # print("existingOutEdge", existingOutEdge, di)
                    n1, n2 = list(existingOutEdge)
                    cnty1, cnty2 = n1[0], n2[0]
                    if n1[1] == 0:
                        cnty1 = fineGraph.nodes[n1[0]][topDesc]
                    if n2[1] == 0:
                        cnty2 = fineGraph.nodes[n2[0]][topDesc]
                    othNode = set([(cnty1, 1), (cnty2, 1)]) 
                    othNode -= set([nodeWithOutEdge])
                    othNode = list(othNode)[0]
                    # print("othNode", othNode, othNode in newDists[di])
                    if othNode in newDists[di]:
                        newDists[di][othNode][2].add(existingOutEdge)
        
        # for d in newDists:
        #     if ("iredell", 1) in newDists[d]:
        #         print("iredell outedges middle", d, newDists[d][("iredell", 1)][2])

        # if print_huh:
        #     print("cntyLevel", cntyLevel)
        #     print("newDists", newDists)
        for di in newDists:
            # print("outEdges[cntyLevel]", di, outEdges[cntyLevel])
            for existingOutEdge in outEdges[cntyLevel]:
                n1, n2 = list(existingOutEdge)
                top1, top2 = n1[0], n2[0]
                if n1[1] == 0:
                    top1 = fineGraph.nodes[n1[0]][topDesc]
                if n2[1] == 0:
                    top2 = fineGraph.nodes[n2[0]][topDesc]
                othCnty = set([(top1, 1), (top2, 1)]) - set([cntyLevel])
                othCnty = list(othCnty)[0]
                # print(top1, top2, othCnty, newDists[di]["top"][0].nodes)
                if othCnty + (di,) in newDists[di]["top"][0].nodes:
                    newDists[di][cntyLevel][2].add(existingOutEdge)
                    if n1[1]==n2[1] and n2[1]==0:
                        # print("newDists[di].keys()", othCnty, newDists[di].keys())
                        newDists[di][othCnty][2].add(existingOutEdge)
        # print("outEdges", outEdges)
        # print("newDists", newDists)

        # for d in newDists:
        #     if ("iredell", 1) in newDists[d]:
        #         print("iredell outedges after", d, newDists[d][("iredell", 1)][2])

        # try:
        #     print("outedges new iredell11", newDists[6][("iredell", 1)][2])
        #     print_huh = True
        # except:
        #     pass

        for di in newDists:
            # print("district", di, newDists[di])
            for cntyWExpTree in newDists[di]:
                if cntyWExpTree == "top":
                    continue
                existingOutEdges = newDists[di][cntyWExpTree][2]
                topExistingEdgeNodes = set([])
                for eOEdge in existingOutEdges:
                    n1, n2 = list(eOEdge)
                    if n1[1] == 0:
                        n1 = (fineGraph.nodes[n1[0]][topDesc], 1)
                    if n2[1] == 0:
                        n2 = (fineGraph.nodes[n2[0]][topDesc], 1)
                    topExistingEdgeNodes.add(n1)
                    topExistingEdgeNodes.add(n2)
                # if print_huh:
                #     print("cntyWExpTree", di, cntyWExpTree, existingOutEdges, topExistingEdgeNodes)
                # print("cntyWExpTree", cntyWExpTree, topExistingEdgeNodes)
                for nbr in newDists[di]["top"][0].neighbors(cntyWExpTree + (di,)):
                    nbr = nbr[:2]
                    if nbr in topExistingEdgeNodes:
                        continue
                    # print("edge to create", cntyWExpTree, nbr)
                    cut2 = False
                    if nbr in newDists[di]:
                        cut2 = True
                    edge = mlg.setCrossEdge(state, cntyWExpTree, nbr, dists, 
                                            info, cut12 = {1: True, 2: cut2}, 
                                            updateDist = newNodeAssignment)
                    # if print_huh:
                    #     print("adding edge w nbr", di, edge, nbr)
                    newDists[di][cntyWExpTree][2].add(edge)
                    n1, n2 = list(edge)
                    cnty1, cnty2 = n1[0], n2[0]
                    if n1[1] == 0:
                        cnty1 = fineGraph.nodes[n1[0]][topDesc]
                    if n2[1] == 0:
                        cnty2 = fineGraph.nodes[n2[0]][topDesc]
                    othNode = set([(cnty1, 1), (cnty2, 1)]) 
                    othNode -= set([cntyWExpTree])
                    othNode = list(othNode)[0]
                    # print("othNode", othNode, othNode in newDists[di])
                    if othNode in newDists[di]:
                        # if print_huh:
                        #     print("OTHER adding edge w nbr", di, edge, nbr, othNode)
                        newDists[di][othNode][2].add(edge)
        # try:
        #     print("outedges new iredell333", newDists[6][("iredell", 1)][2])
        # except:
        #     pass

        # update persistent edge possibilities
        d1, d2 = list(dists)
        noPossibleEdges = 0
        # print("handling persistentEdge")
        for node in newDistAssignment[d1][0][cntyLevel[0]]:
            for nbr in fineGraph.neighbors(node):
                topNbr = fineGraph.nodes[nbr][topDesc]
                if topNbr != cntyLevel[0]:
                    continue
                newNbrDist = newNodeAssignment[cntyLevel[0]][1][nbr]
                if newNbrDist != d1:
                    noPossibleEdges += 1

    cn1, cn2 = list(cutEdge)
    newPersistentEdge = ((cn1, cn2), noPossibleEdges, (0, 0))
    toReturn = (newDists, newDistAssignment, newNodeAssignment, 
                newPersistentEdge, cutEdges)
    return toReturn

################################################################################

def msMergeSplit(state, info):
    '''Returns a single merge-split step with associated probability,
       not including spanning trees.'''
    newdistricts = {}
    rng = info["rng"]
    # print(rng.uniform(0,1))
    persistentEdges = state["layeredGraph"]["persistentEdges"]
    persistentEdge = rng.choice(sorted(list(persistentEdges)))
########
    # fineGraph = state["layeredGraph"]["graphLevels"][0]
    # print("persistentEdge cnty", 
    #       fineGraph.nodes[persistentEdge[0][0]]["County"])
    # persistentEdge = (('2653', '2667'), 23, (0, 0))
    # persistentEdge = (('434', '462'), 6, (0, 0))
    # persistentEdge = (('113', '122'), 14, (0, 0))
    # persistentEdge = (('1703', '1730'), 5, (0, 0))
    # persistentEdge = (('2259', '2304'), 44, (0, 0))
    # print("persistentEdge", persistentEdge)
    # print("DEBUG ON -- enforcing persistentEdge")
########
    d1, d2 = parsePersistentEdgeDists(state, persistentEdge)

    #propose top tree
    # print("buildMergedTopTree")
    mergedData = mlg.buildMergedTopTree(state, info, d1, d2, persistentEdge)
    mergedTree, mergedGraph, mergedPop = mergedData
    # print("finished buildMergedTopTree")

    # print(state["layeredGraph"]["nestedTrees"][d1]["top"][0].edges)
    # print(mergedTree.edges)

    # print("before proposeCut")
    # graph = state["layeredGraph"]["topDistGraph"]
    topLevel = len(state["layeredGraph"]["hierarchy"])-1
    proposalData = proposeCut(mergedTree, mergedPop, mergedGraph, state, info, 
                              set([d1, d2]), persistentEdge, topLevel)
    newDists, newDistAssignments, newNodeAssignments, newPerstEdg, cutEdges =\
                                                                    proposalData

    # print("after proposeCut")
    # print("newDists", newDists)
    # print("newDists", newDists)
    # print("newDistAssignments", newDistAssignments)
    # print("newNodeAssignments", newNodeAssignments)
    # print("newPerstEdg", newPerstEdg)
    # print("cutEdges", cutEdges)

    prstntEdgeMap = {persistentEdge : newPerstEdg}
    if len(cutEdges) == 0:
        return (proposalData, prstntEdgeMap, 0)

    pEDists = set([])
    for pE in persistentEdges:
        if pE == persistentEdge:
            continue
        n1, n2 = pE[0]
        d1 = mlg.getDistricts(state, n1, 0, updateDist = newNodeAssignments)
        d2 = mlg.getDistricts(state, n2, 0, updateDist = newNodeAssignments)
        dists = frozenset([d1, d2])
        # c1 = state["layeredGraph"]["graphLevels"][0].nodes[n1]["County"]
        # c2 = state["layeredGraph"]["graphLevels"][0].nodes[n2]["County"]
        # print("d1, d2, cnty1, cnty2", (d1, c1), (d2, c2), pE == persistentEdge)
        if dists in pEDists:
            return (proposalData, prstntEdgeMap, 0)
        pEDists.add(dists)
    n1, n2 = newPerstEdg[0]
    d1 = mlg.getDistricts(state, n1, 0, updateDist = newNodeAssignments)
    d2 = mlg.getDistricts(state, n2, 0, updateDist = newNodeAssignments)
    dists = frozenset([d1, d2])
    if dists in pEDists:
        return (proposalData, prstntEdgeMap, 0)

    # print("persistentEdges", persistentEdges)
    # print("pEDists", pEDists)
    # print("dists", dists)

    oldEdgeCuts = mlg.countEdgeCuts(state, d1, d2, persistentEdge, info)
    # print("oldEdgeLen", len(oldEdgeCuts))
    p = len(cutEdges)/len(oldEdgeCuts)

    return (proposalData, prstntEdgeMap, p)


def msMergeSplitGamma(gamma):
    def f(state, info):
        '''Returns a single merge-split step with associated probability,
           including spanning trees.'''
        (proposalData, persistentEdge, p) = msMergeSplit(state, info)
        # if p:

            # graph = state["graph"]
            # nD1 = newdistricts["districts"][d1]
        #     nD2 = newdistricts["districts"][d2]
        #     spTrCnt1 = tree.nspanning(nD1)
        #     spTrCnt2 = tree.nspanning(nD2)
        #     newdistricts["spanningTreeCounts"] = {d1 : spTrCnt1, d2 : spTrCnt2}
        #     spTrCnt1Old = state["spanningTreeCounts"][d1]
        #     spTrCnt2Old = state["spanningTreeCounts"][d2]
        #     p *= exp(gamma*(spTrCnt1Old + spTrCnt2Old - spTrCnt1 - spTrCnt2))
        return (proposalData, persistentEdge, p)
    return f

def define(mcmcArgs):
    gamma = mcmcArgs["parameters"]["gamma"]
    mcmcArgs["proposal"] = "multiScaleMergeSplitFine"
    if gamma == 0:
        return msMergeSplit, mcmcArgs
    else:
        return msMergeSplitGamma(gamma), mcmcArgs
