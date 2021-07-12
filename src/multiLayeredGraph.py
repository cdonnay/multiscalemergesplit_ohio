import copy
import networkx as nx
import os

import tree

import importlib

importlib.reload(tree)

################################################################################

def buildNestedGraphs(state):
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    # state["layeredGraph"]["edgeCounts"] = {}
    baseGraph = state["graph"]
    nestedGraphs = {0 : baseGraph}
    # baseEdgeCounts = {e : 1 for e in state["graph"].edges}
    for ii in range(1, len(nestedHierarchy)):
        level = nestedHierarchy[ii]
        # G = nx.Graph()
        G = nx.MultiGraph()
        edgeCounts = {}
        nodes = set([baseGraph.nodes[n][level] for n in baseGraph.nodes])
        G.add_nodes_from(nodes)
        for node in nodes:
            baseNodes = set([n for n in baseGraph.nodes 
                             if baseGraph.nodes[n][level] == node])
            population = sum([baseGraph.nodes[n]["Population"] 
                              for n in baseNodes])
            G.nodes[node]["nestedElements"] = baseNodes
            G.nodes[node]["Population"] = population
            for jj in range(ii+1, len(nestedHierarchy)):
                levelUp = nestedHierarchy[jj]
                egNode = list(baseNodes)[0]
                G.nodes[node][levelUp] = baseGraph.nodes[egNode][levelUp]
        for e in baseGraph.edges:
            cnty1 = baseGraph.nodes[e[0]][level]
            cnty2 = baseGraph.nodes[e[1]][level]
            if cnty1 != cnty2:
                newEdge = (min(cnty1, cnty2), max(cnty1, cnty2))
                # if newEdge in G.edges:
                #     edgeCounts[newEdge] += 1
                # else:
                #     G.add_edge(newEdge[0], newEdge[1])
                #     edgeCounts[newEdge] = 1
                G.add_edge(newEdge[0], newEdge[1])

        nestedGraphs[ii] = G
        # state["layeredGraph"]["edgeCounts"][level] = edgeCounts
        baseGraph = G
        # baseEdgeCountsGraph = edgeCounts
    return nestedGraphs

################################################################################

def getNbrsAtAltLayer(coarseGraph, cNode, gap, fineInd):
    if gap != 1:
        nbrs = coarseGraph.nodes[cNode]["downNbrs"][fineInd+1]
    else:
        nbrs = coarseGraph.neighbors(cNode)
    return nbrs                                      

################################################################################

def setDownNbrs(state, downNbrs, cNode, cLevel, nbr, nLevel):
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    coarseGraph = state["layeredGraph"]["graphLevels"][cLevel]
    fineGraph = state["layeredGraph"]["graphLevels"][nLevel]

    cDesc = nestedHierarchy[cLevel]
    # print("cDesc", cDesc)

    if nLevel == cLevel:
        edges = coarseGraph.number_of_edges(cNode, nbr)
    else:
        edges = 0
        for nnbr in fineGraph.neighbors(nbr):
            if fineGraph.nodes[nnbr][cDesc] == cNode:
                edges += fineGraph.number_of_edges(nnbr, nbr)

    # print("setDownNbrs", cNode, cLevel, nbr, nLevel, edges)

    if nLevel == 0:
        downNbrs[nbr] = [edges, {}]
    else:
        ddownNbrs = {}
        finerGraph = state["layeredGraph"]["graphLevels"][nLevel-1]

        # print("nbr", nbr)
        # print("nested elements of nbr", fineGraph.nodes[nbr]["nestedElements"])
        for nestedNode in fineGraph.nodes[nbr]["nestedElements"]:
            coarseNeighbor = False
            for nstNbr in finerGraph.neighbors(nestedNode):
                if finerGraph.nodes[nstNbr][cDesc] == cNode:
                    coarseNeighbor = True
            if coarseNeighbor:
                ddownNbrs = setDownNbrs(state, ddownNbrs, cNode, cLevel, 
                                        nestedNode, nLevel-1)
        downNbrs[nbr] = [edges, ddownNbrs]
    # print("setDownNbrs returnning")

    return downNbrs

################################################################################

def getDistricts(state, node, level, updateDist = {}):
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = max(state["layeredGraph"]["hierarchy"])
    g = state["layeredGraph"]["graphLevels"][level]


    if level == topLevel:
        try:
            dists = updateDist[node]
            # print("through cnty try 1")
        except:
            dists = state["districts"]["n2d"][node]
        if isinstance(dists, int):
            return dists
        else:
            return dists[0]

    try:
        lU = updateDist
        for l in range(topLevel, level-1, -1):
            if l != level:
                lDesc = g.nodes[node][nestedHierarchy[l]]
            else:
                lDesc = node
            dists = lU[lDesc]
            if isinstance(dists, int):
                return dists
            else:
                lU = dists[1]
            # print("through pct try 0")
    except:
        lU = state["districts"]["n2d"]
        for l in range(topLevel, level-1, -1):
            if l != level:
                lDesc = g.nodes[node][nestedHierarchy[l]]
            else:
                lDesc = node
            dists = lU[lDesc]
            if isinstance(dists, int):
                return dists
            else:
                lU = dists[1]

    if isinstance(dists, int):
        return dists
    else:
        return dists[0]

################################################################################

def getSplitNodes(state, level, node = "", dist = ""):

    topLevel = max(state["layeredGraph"]["hierarchy"])
    if level == topLevel and node != "":
        raise Exception ("Cannot find split nodes of node", node, 
                         "at top level")

    if node == "":
        nodes = state["layeredGraph"]["graphLevels"][level].nodes
    else:    
        g = state["layeredGraph"]["graphLevels"][level+1]
        nodes = g.nodes[node]["nestedElements"]

    splitNodes = set([])

    for node in nodes:
        dists = getDistricts(state, node, level)
        if not isinstance(dists, int):
            if dist == "":
                splitNodes.add(node)
            elif dist in dists:
                # print(node, dist, dists)
                splitNodes.add(node)
        # print(node, dists)
    return splitNodes

################################################################################

def buildCrossHeirarchyEdges(state):
    layeredGraph = state["layeredGraph"]
    baseGraph = layeredGraph["graphLevels"][0]
    nestedHierarchy = layeredGraph["hierarchy"]

    for level in range(len(nestedHierarchy)):
        coarseGraph = layeredGraph["graphLevels"][level]
        for cNode in coarseGraph.nodes:
            # if cNode != "jones":
            #     continue
            # print("coarse node", cNode)
            coarseGraph.nodes[cNode]["downNbrs"] = {}
            for cNbr in coarseGraph.neighbors(cNode):
                # print("coarse neighbor", cNbr)
                downNbrs = {}
                # print("calling set down nbrs", cNbr, level, cNode, level)
                downNbrs = setDownNbrs(state, downNbrs, cNode, level, 
                                       cNbr, level)
                coarseGraph.nodes[cNode]["downNbrs"].update(downNbrs)
    return state

################################################################################

def assignDistricts(state, initPlanKey):
    nestedHierarchy = state["layeredGraph"]["hierarchy"]

    allDists = set([])

    fineGLayer = state["layeredGraph"]["graphLevels"][0]
    for node in fineGLayer:
        dist = int(fineGLayer.nodes[node][initPlanKey])
        fineGLayer.nodes[node]["dist"] = [dist]
        allDists.add(dist)
    state["noDists"] = len(allDists)

    for layer in range(1, len(nestedHierarchy)):
        GLayer = state["layeredGraph"]["graphLevels"][layer]
        GfLayer = state["layeredGraph"]["graphLevels"][layer-1]
        for node in GLayer.nodes:
            dists = set()
            for fNode in GLayer.nodes[node]["nestedElements"]:
                dists = dists.union(GfLayer.nodes[fNode]["dist"])
            GLayer.nodes[node]["dist"] = list(dists)

    return state


################################################################################

# def getTopLinks(state, nbr, distToLinkCount, (node, level)):
#     nestedHierarchy = state["layeredGraph"]["hierarchy"]
#     topLevel = len(nestedHierarchy)-1
#     topGraph = state["layeredGraph"]["graphLevels"][topLevel]
#     levelGraph = state["layeredGraph"]["graphLevels"][level]

#     nodeDists = levelGraph.nodes[node]["dist"]
    
    # if len(nodeDists) == 1:
    #     dist = list(nodeDists)[0]
    #     if topLevel == level:
    #         distToLinkCount[dist] += topGraph.number_of_edges(node, nbr)
    #     else:
    #         nodeLevelDesc = nestedHierarchy[level]
    #         topNode = topGraph.nodes[nbr]
    #         noCrossNodes = topNode['splitLvlNbrs'][nodeLevelDesc][node]
    #         distToLinkCount[dist] += noCrossNodes
    # else:
    #     # get level-1 neighbors; for each neighbor, run this function
    #     downLevel = level-1
    #     nodeLevelDesc = nestedHierarchy[downLevel]
    #     topNode = topGraph.nodes[nbr]
    #     crossNodes = topNode['splitLvlNbrs'][nodeLevelDesc]
    #     for 

    
    # return distToLinkCount


    


################################################################################

def addNestedNodes(graph, state, node, level):
    graph = state["layeredGraph"]["graphLevels"][level]
    dists = graph.nodes[node]["dist"]
    if len(dists) == 1:
        graph.add_node(node)
        graph.nodes[node]["level"] = level
    else:
        nestedNodes = graph.nodes[node]["nestedElements"]
        for nestedNode in nestedNodes:
            lm1 = level-1
            graph = addNestedNodes(graph, state, nestedNode, lm1)
    return graph

################################################################################

def fixEdges(graph, state, node, level):
    graph = state["layeredGraph"]["graphLevels"][level]
    dists = graph.nodes[node]["dist"]
    if len(dists) == 1:
        graph.add_node(node)
        graph.nodes[node]["level"] = level
    else:
        nestedNodes = graph.nodes[node]["nestedElements"]
        for nestedNode in nestedNodes:
            lm1 = level-1
            graph = addNestedNodes(graph, state, nestedNode, lm1)

    return mixedGraph

################################################################################

def getTopLevelNodes(state, dist):
    nestedHeirarchy = state["layeredGraph"]["hierarchy"]
    topLevel = len(nestedHeirarchy)-1
    topDesc = nestedHeirarchy[topLevel]

    topDistGraph = state["layeredGraph"]["topDistGraph"]
    nodes = set([])
    for node in state["districts"]["d2n"][dist][topLevel]:
        nodes.add((node, topLevel, dist))

    return nodes

################################################################################

def findSingleDistDownNbrs(state, node, nbrEdgeDict, baseLevel, 
                           lvlToSDNbr = {}):
    baseGraph = state["layeredGraph"]["graphLevels"][baseLevel]
    dists = baseGraph.nodes[node]['dist']
    if len(dists) == 1:
        if baseLevel in lvlToSDNbr:
            lvlToSDNbr[baseLevel].add(node)
        else:
            lvlToSDNbr[baseLevel] = set([node])
    else:
        for subNode in nbrEdgeDict[node][1]:
            lvlToSDNbr = findSingleDistDownNbrs(state, subNode, 
                                                nbrEdgeDict[node][1], 
                                                baseLevel-1, lvlToSDNbr)
    return lvlToSDNbr


################################################################################

def findDistConnections(state, node, nodeDist, nodeLevel, nbrEdgeDict, 
                        nbrLevel, distPairToEdges):
    for nbr in nbrEdgeDict:
        nodeGraph = state["layeredGraph"]["graphLevels"][nbrLevel]
        dists = nodeGraph.nodes[nbr]['dist']
        if len(dists) == 1:
            othrDist = list(dists)[0]
            key = (nodeDist, othrDist)
            if key in distPairToEdges:
                distPairToEdges[key] += nbrEdgeDict[nbr][0]
            else:
                distPairToEdges[key] = nbrEdgeDict[nbr][0]
        else:
            distPairToEdges = findDistConnections(state, node, nodeDist, 
                                                  nodeLevel, 
                                                  nbrEdgeDict[nbr][1], 
                                                  nbrLevel-1, 
                                                  distPairToEdges)
    return distPairToEdges


################################################################################

def findDistToEdgeCounts(state, sdNode, distToEdgeCount, nodeEdgeDict, level,
                         topNbr, topDesc):
    nodeGraph = state["layeredGraph"]["graphLevels"][level]
    dnNbrs = nodeGraph.nodes[sdNode]['downNbrs']
    nodeDist = list(nodeGraph.nodes[sdNode]['dist'])[0]
    nbrEdgeDict = { k : dnNbrs[k] for k in dnNbrs 
                    if nodeGraph.nodes[k][topDesc] == topNbr }
    # print(sdNode)
    # print(nbrEdgeDict)
    distPairToEdges = findDistConnections(state, sdNode, nodeDist, level, 
                                          nbrEdgeDict, level, {}) 
    # print(distPairToEdges)
    for dp in distPairToEdges:
        if dp in distToEdgeCount:
            distToEdgeCount[dp] += distPairToEdges[dp]
        else:
            distToEdgeCount[dp] = distPairToEdges[dp]
    return distToEdgeCount

################################################################################

def getLowestPreservedLevel(state, nodes, level):
    
    g = state["layeredGraph"]["graphLevels"][level]
    try:
        gm1 = state["layeredGraph"]["graphLevels"][level-1]
    except:
        raise Exception("nodes", nodes, "cannot be split; no lower level")
    
    noDists = 0
    splitNodes = set([])

    for node in nodes:
        for subNode in g.nodes[node]["nestedElements"]:
            noNodeDists = len(gm1.nodes[subNode]['dist'])
            noDists = max(noDists, noNodeDists)
            if noNodeDists > 1:
                splitNodes.add(subNode)
    if noDists > 1:
        level, splitNodes = getLowestPreservedLevel(state, splitNodes, level-1)
    if len(splitNodes) == 0:
        splitNodes = nodes

    return level-1, splitNodes

################################################################################

def findNestedPopulationsByDist(state, node, level):
    graph = state["layeredGraph"]["graphLevels"][level]
    dists = graph.nodes[node]["dist"]

    popsByDist = {d : 0 for d in dists}
    nodesByDist = {d : {l : set([]) for l in range(level+1)} for d in dists}

    if len(dists) == 1:
        d = list(dists)[0]
        popsByDist[d] = graph.nodes[node]["Population"]
        nodesByDist[d][level].add(node)
    else:
        for nn in graph.nodes[node]["nestedElements"]:
            popsByDistLow, nodesByDistLower = \
                                 findNestedPopulationsByDist(state, nn, level-1)
            for d in popsByDistLow:
                popsByDist[d] += popsByDistLow[d]
                for l in nodesByDistLower[d]:
                    nodesByDist[d][l].update(nodesByDistLower[d][l])

    return popsByDist, nodesByDist

################################################################################

def buildTopDistGraph(state):
    topDistGraph = nx.MultiGraph()
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = len(nestedHierarchy)-1

    topGraph = state["layeredGraph"]["graphLevels"][topLevel]
    topDistGraph = copy.deepcopy(topGraph)

    nodeMap = {n : (n, topLevel, topGraph.nodes[n]["dist"][0]) 
               for n in topDistGraph.nodes}
    topDistGraph = nx.relabel_nodes(topDistGraph, nodeMap)

    splitTopNodes = set([])

    for node in topGraph.nodes:
        dists = topGraph.nodes[node]["dist"]
        if len(dists) > 1:
            topDistGraph.remove_node((node, topLevel, dists[0]))
            # topDistGraph.remove_node(node)
            popsByDist, nodesByDist = \
                              findNestedPopulationsByDist(state, node, topLevel)
            # if node == 'wilson':
            #     print(node, dists, popsByDist)

            for d in popsByDist:
                node1 = (node, topLevel, d)
                # node1 = (node, d)
                topDistGraph.add_node(node1)
                topDistGraph.nodes[node1]['dist'] = [d]
                topDistGraph.nodes[node1]['Population'] = popsByDist[d]
                topDistGraph.nodes[node1]['nestedElements'] = \
                                 {l : nodesByDist[d][l] for l in nodesByDist[d]}

            for nbr in topGraph.neighbors(node):
                # print(node, nbr)
                # print("node dists", topGraph.nodes[node]['dist'])
                # print("nbr dists", topGraph.nodes[nbr]['dist'])

                nbrEdgeDict = topGraph.nodes[nbr]['downNbrs']
                # print(nbrEdgeDict)
                sdNodes = findSingleDistDownNbrs(state, node, nbrEdgeDict, 
                                                 topLevel, {})
                # print("sdNodes", sdNodes)
                nodeEdgeDict = topGraph.nodes[node]['downNbrs']
                # print(nodeEdgeDict)
                distToEdgeCount = {}
                for level in sdNodes:
                    for sdNode in sdNodes[level]:
                        # print("calling findDistToEdgeCounts", level, sdNode)
                        distToEdgeCount = findDistToEdgeCounts(state, sdNode, 
                                                               distToEdgeCount, 
                                                               nodeEdgeDict,
                                                               level, 
                                                               nbr, "County")

                nbrDists = topGraph.nodes[nbr]['dist']
                for dpair in distToEdgeCount:
                    node1 = (node, topLevel, dpair[0])
                    node2 = (nbr, topLevel, dpair[1])
                    # print("node2", node2)    
                    if node2 not in topDistGraph.nodes:
                        topDistGraph.add_node(node2)
                        topDistGraph.nodes[node2]['dist'] = [dpair[1]]
                    for ii in range(distToEdgeCount[dpair]):
                        topDistGraph.add_edge(node1, node2)
            splitTopNodes.add(node)
    state["layeredGraph"]["topDistGraph"] = topDistGraph
    state["layeredGraph"]["splitTopNodes"] = splitTopNodes
    return state

################################################################################

def buildTopDistSubGraph(state, dist, updateDistAsgn = {}, updateNodeAsgn = {}):
    topDistSubGraph = nx.MultiGraph()
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = len(nestedHierarchy)-1

    cntys = updateDistAsgn[1]

    topGraph = state["layeredGraph"]["graphLevels"][topLevel]
    topDistSubGraph = copy.deepcopy(topGraph.subGraph(cntys))

    nodeMap = {n : (n, topLevel) for n in topDistSubGraph.nodes}
    topDistGraph = nx.relabel_nodes(topDistSubGraph, nodeMap)

    splitTopNodes = set([])

    for node in cntys:
        dists = updateNodeAsgn[node]
        if isinstance(dists, int):
            dists = [dists]
        if len(dists) > 1:
            topDistGraph.remove_node((node, topLevel))
            
            
            # if node == 'wilson':
            #     print(node, dists, popsByDist)

            for d in popsByDist:
                node1 = (node, topLevel, d)
                # node1 = (node, d)
                topDistGraph.add_node(node1)
                topDistGraph.nodes[node1]['dist'] = [d]
                topDistGraph.nodes[node1]['Population'] = popsByDist[d]
                topDistGraph.nodes[node1]['nestedElements'] = \
                                 {l : nodesByDist[d][l] for l in nodesByDist[d]}

            for nbr in topGraph.neighbors(node):
                # print(node, nbr)
                # print("node dists", topGraph.nodes[node]['dist'])
                # print("nbr dists", topGraph.nodes[nbr]['dist'])

                nbrEdgeDict = topGraph.nodes[nbr]['downNbrs']
                # print(nbrEdgeDict)
                sdNodes = findSingleDistDownNbrs(state, node, nbrEdgeDict, 
                                                 topLevel, {})
                # print("sdNodes", sdNodes)
                nodeEdgeDict = topGraph.nodes[node]['downNbrs']
                # print(nodeEdgeDict)
                distToEdgeCount = {}
                for level in sdNodes:
                    for sdNode in sdNodes[level]:
                        # print("calling findDistToEdgeCounts", level, sdNode)
                        distToEdgeCount = findDistToEdgeCounts(state, sdNode, 
                                                               distToEdgeCount, 
                                                               nodeEdgeDict,
                                                               level, 
                                                               nbr, "County")

                nbrDists = topGraph.nodes[nbr]['dist']
                for dpair in distToEdgeCount:
                    node1 = (node, topLevel, dpair[0])
                    node2 = (nbr, topLevel, dpair[1])
                    # print("node2", node2)    
                    if node2 not in topDistGraph.nodes:
                        topDistGraph.add_node(node2)
                        topDistGraph.nodes[node2]['dist'] = [dpair[1]]
                    for ii in range(distToEdgeCount[dpair]):
                        topDistGraph.add_edge(node1, node2)
            splitTopNodes.add(node)
    state["layeredGraph"]["topDistGraph"] = topDistGraph
    state["layeredGraph"]["splitTopNodes"] = splitTopNodes
    return state

################################################################################

def buildSublevelDistGraph(state, node, level, dist):

    graph = state["layeredGraph"]["graphLevels"][level]
    fineGraph = state["layeredGraph"]["graphLevels"][level-1]

    # print(nx.is_forzen(graph))
    # print(nx.is_forzen(finegraph))

    nodes = graph.nodes[node]["nestedElements"]
    subGraph = fineGraph.subgraph(set(graph.nodes[node]["nestedElements"]))
    # subGraph = nx.MultiGraph(subGraph)
    subGraph = nx.Graph(subGraph)

    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = max(nestedHierarchy)
    dists = getDistricts(state, node, level)

    if isinstance(dists, int):
        if dists == dist:
            return subGraph
        else:
            return nx.Graph()
    # print(dist, subGraph.nodes)
    levelGraph = copy.deepcopy(subGraph)
    
    # print("nodes", nodes)
    for node in nodes:
        dists = getDistricts(state, node, level-1)
        # print(node, dists, dist)
        if isinstance(dists, int):
            if dists != dist:
                levelGraph.remove_node(node)
        elif dist not in dists:
            levelGraph.remove_node(node)
        '''elif len(dists) > 1:
            raise Exception("Warning -- not yet sussed out")
            levelGraph.remove_node(node)
            popsByDist, nodesByDist = \
                              findNestedPopulationsByDist(state, node, level-1)
            # if node == 'wilson':
            #     print(node, dists, popsByDist)

            for d in popsByDist:
                node1 = (node, topLevel, d)
                # node1 = (node, d)
                topDistGraph.add_node(node1)
                topDistGraph.nodes[node1]['dist'] = [d]
                topDistGraph.nodes[node1]['Population'] = popsByDist[d]
                topDistGraph.nodes[node1]['nestedElements'] = \
                                 {l : nodesByDist[d][l] for l in nodesByDist[d]}

            for nbr in topGraph.neighbors(node):
                # print(node, nbr)
                # print("node dists", topGraph.nodes[node]['dist'])
                # print("nbr dists", topGraph.nodes[nbr]['dist'])

                nbrEdgeDict = topGraph.nodes[nbr]['downNbrs']
                # print(nbrEdgeDict)
                sdNodes = findSingleDistDownNbrs(state, node, nbrEdgeDict, 
                                                 topLevel, {})
                # print("sdNodes", sdNodes)
                nodeEdgeDict = topGraph.nodes[node]['downNbrs']
                # print(nodeEdgeDict)
                distToEdgeCount = {}
                for level in sdNodes:
                    for sdNode in sdNodes[level]:
                        # print("calling findDistToEdgeCounts", level, sdNode)
                        distToEdgeCount = findDistToEdgeCounts(state, sdNode, 
                                                               distToEdgeCount, 
                                                               nodeEdgeDict,
                                                               level, 
                                                               nbr, "County")

                nbrDists = topGraph.nodes[nbr]['dist']
                for dpair in distToEdgeCount:
                    node1 = (node, topLevel, dpair[0])
                    node2 = (nbr, topLevel, dpair[1])
                    # print("node2", node2)    
                    if node2 not in topDistGraph.nodes:
                        topDistGraph.add_node(node2)
                        topDistGraph.nodes[node2]['dist'] = [dpair[1]]
                    for ii in range(distToEdgeCount[dpair]):
                        topDistGraph.add_edge(node1, node2)
            splitTopNodes.add(node)#'''
    # print(levelGraph.nodes)
    return levelGraph
################################################################################

def setCrossEdge(state, e1s, e2s, dists, info, cut12 = {}, updateDist = {}):
    node1, level1 = e1s
    node2, level2 = e2s
    level = max(level1, level2)

    coarseGraph = state["layeredGraph"]["graphLevels"][level]
    fineGraph = state["layeredGraph"]["graphLevels"][level-1]
    coarseDesc = state["layeredGraph"]["hierarchy"][level]

    coarse1 = node1
    coarse2 = node2
    if level1 < level:
        coarse1 = fineGraph.nodes[node1][coarseDesc] 
    if level2 < level:
        coarse2 = fineGraph.nodes[node2][coarseDesc] 

    nbrEdgeDict = coarseGraph.nodes[coarse2]['downNbrs'][coarse1][1]
    # print(node, nbrName, nbrEdgeDict)
    subNodes = set([])
    for sNodeNextToNbr in nbrEdgeDict:
        nodeDists = getDistricts(state, sNodeNextToNbr, level-1, updateDist)
        if isinstance(nodeDists, int):
            commonDists = set([nodeDists])
        else:
            commonDists = set(nodeDists)
        commonDists = commonDists.intersection(dists)
        if len(commonDists) == 0:
            continue
        for sNbr in fineGraph.neighbors(sNodeNextToNbr):
            nodeDists = getDistricts(state, sNbr, level-1, updateDist)
            if isinstance(nodeDists, int):
                commonDists = set([nodeDists])
            else:
                commonDists = set(nodeDists)
            commonDists = commonDists.intersection(dists)
            if fineGraph.nodes[sNbr][coarseDesc]==coarse2 and \
               len(commonDists) > 0:
               subNodes.add((sNodeNextToNbr, sNbr))
    
    rm = set([])
    if level1 != level:
        for e in subNodes:
            if e[0] != node1:
                rm.add(e)
    if level2 != level:
        for e in subNodes:
            if e[1] != node2:
                rm.add(e)
    subNodes -= rm

    subNode = info['rng'].choice(sorted(subNodes))

    try:
        cut1 = cut12[1]
    except: 
        cut1 = not isinstance(getDistricts(state, node1, level, updateDist), 
                              int)
    try:
        cut2 = cut12[2]
    except:
        cut2 = not isinstance(getDistricts(state, node2, level, updateDist), 
                               int)

    if cut1 and cut2:
        edge = frozenset([(subNode[0], level-1), (subNode[1], level-1)])
    elif cut1:
        edge = frozenset([(subNode[0], level-1), (node2, level)])
    elif cut2:
        edge = frozenset([(node1, level), (subNode[1], level-1)])
    else:
        edge = frozenset([(node1, level), (node2, level)])

    return edge

################################################################################

def findPersistentEdges(state, lowLevel, lowSplitNodes):
    gT = state["layeredGraph"]["graphLevels"][lowLevel + 1]
    g = state["layeredGraph"]["graphLevels"][lowLevel]

    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    tDesc = nestedHierarchy[lowLevel + 1]

    edges = {}
    for n in lowSplitNodes:
        for node in gT.nodes[n]["nestedElements"]:
            for nbr in g.neighbors(node):
                # print(n, node, nbr, tdesc, g.nodes[nbr][tDesc], g.nodes[node][tDesc])
                if g.nodes[nbr][tDesc] != g.nodes[node][tDesc]:
                    continue
                d1 = g.nodes[node]["dist"][0]
                d2 = g.nodes[nbr]["dist"][0]
                if d1 == d2:
                    continue
                edges[(min(node, nbr), max(node, nbr))] = \
                                                    g.number_of_edges(node, nbr)
    return edges


################################################################################

def initializePersistentEdges(state):
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = len(nestedHierarchy)-1
    topGraph = state["layeredGraph"]["graphLevels"][topLevel]
    fineGraph = state["layeredGraph"]["graphLevels"][topLevel-1]

    splitTopNodes = state["layeredGraph"]["splitTopNodes"]
    persistentEdges = set([])
    linkedDists = set([])

    for node in splitTopNodes:
        dists = topGraph.nodes[node]["dist"]
        lowLevel, lowSplitNodes = getLowestPreservedLevel(state, set([node]), 
                                                          topLevel)
        edges = findPersistentEdges(state, lowLevel, lowSplitNodes)
        edge = sorted(list(edges.keys()))[0]
        noEdges = sum([edges[e] for e in edges])
        persistentEdges.add(((edge[0], edge[1]), noEdges, (lowLevel, lowLevel)))
        linkedDists.add(frozenset(dists))

    moreEdges = state["maxCntySplts"] - len(persistentEdges)
    # print(splitTopNodes)
    # print(len(splitTopNodes))
    # print("persistentEdges", persistentEdges)
    # print("persistentEdges length", len(persistentEdges))
    # print(linkedDists)
    # print("adding", moreEdges, "more edges")
    for ii in range(moreEdges):
        success = False
        for n in topGraph.nodes:
            if success:
                break
            ndists = topGraph.nodes[n]["dist"]
            if len(ndists) > 1:
                continue
            ndist = ndists[0]
            for nbr in topGraph.neighbors(n):
                nbrdists = topGraph.nodes[nbr]["dist"]
                if len(nbrdists) > 1:
                    continue
                nbrdist = nbrdists[0]
                if ndist == nbrdist:
                    continue
                noEdges = sum([v[0] for v in topGraph.nodes[nbr]["downNbrs"][n][1].values()])
                for fineNode in topGraph.nodes[nbr]["downNbrs"][n][1]:
                    for fineNbr in fineGraph.neighbors(fineNode):
                        fineNbrDist = fineGraph.nodes[fineNbr]["dist"][0]
                        if fineNbrDist == ndist:
                            continue
                        if frozenset([fineNbrDist, ndist]) in linkedDists:
                            continue
                        newEdge = ((fineNode, fineNbr), noEdges, (lowLevel, lowLevel))
                        if newEdge in persistentEdges:
                            continue
                        # print("adding edge", fineNode, fineNbr, noEdges)
                        persistentEdges.add(((fineNode, fineNbr), noEdges, (lowLevel, lowLevel)))
                        linkedDists.add(frozenset([fineNbrDist, ndist]))
                        success = True
                        break
                    if success:
                        break
                if success:
                    break
            if success:
                break
        assert(success)
    # print()
    # print("persistentEdges", persistentEdges)
    # print("persistentEdges length", len(persistentEdges))
    # raise Exception("here")
    
    state["layeredGraph"]["persistentEdges"] = persistentEdges 
    return state

################################################################################

def buildMixedGraph(state):
    topDistGraph = nx.MultiGraph()
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = len(nestedHierarchy)-1

    topGraph = state["layeredGraph"]["graphLevels"][topLevel]
    mixedGraph = copy.deepcopy(topGraph)

    for node in topGraph.nodes:
        dists = topGraph.nodes[node]["dist"]
        if len(dists) == 1:
            mixedGraph.nodes[node]["level"] = topLevel
        else:
            subNodes = topLevel
            mixedGraph.remove_node(node)
            mixedGraph = addNestedNodes(mixedGraph, state, node, topLevel)
            mixedGraph = fixEdges(mixedGraph, state, node, topLevel)
    
    state["layeredGraph"]["mixedGraph"] = mixedGraph
    return state

################################################################################

def readTopDistOrientation(state, orientationPath):
    cntyToOrientedNbrs = {}
    oNbrLines = open(orientationPath).readlines()
    for l in oNbrLines:
        splitline = l.rstrip().lower().split("\t")
        keyCnty = splitline[0]
        onbrs = splitline[1:]
        cntyToOrientedNbrs[keyCnty] = onbrs
    state["layeredGraph"]["orientedNeighbors"] = {1 : cntyToOrientedNbrs}
    print("Warning -- this is a hack")
    return state

################################################################################

def buildGraph(state, nestedHierarchy, initPlanKey):
    state["layeredGraph"] = {}
    state["layeredGraph"]["hierarchy"] = nestedHierarchy

    state["layeredGraph"]["graphLevels"] = buildNestedGraphs(state)
    state = buildCrossHeirarchyEdges(state)
    state = assignDistricts(state, initPlanKey)
    state = buildTopDistGraph(state)
    orientationPath = os.path.join("..", "inputData", "NC",
                                   "NC_orientedCntyNeighbors.txt")
    state = readTopDistOrientation(state, orientationPath)
    state = initializePersistentEdges(state)

    return state

################################################################################

def buildDistricting(state, level, nodes):
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = max(nestedHierarchy.keys())

    if level == topLevel:
        state["districts"] = {}
        state["districts"]["d2n"] = {d : {l : {} for l in nestedHierarchy}
                                     for d in range(1, state["noDists"]+1)}
        for d in range(1, state["noDists"]+1):
            state["districts"]["d2n"][d][topLevel] = set([])
        state["districts"]["n2d"] = {}

    g = state["layeredGraph"]["graphLevels"][level]

    for node in nodes:
        # print(node)
        dists = [int(d) for d in g.nodes[node]["dist"]]

        lookUp = state["districts"]["n2d"]
        for l in range(topLevel, level, -1):
            key = g.nodes[node][nestedHierarchy[l]]
            lookUp = lookUp[key][1]

        if len(dists) == 1:
            dist = int(list(dists)[0])
            lookUp[node] = dist

            for l in range(topLevel, level-1, -1):
                try:
                    lookUp = state["districts"]["d2n"][dist][l]
                except Exception:
                    print(state["districts"]["d2n"])
                    print("dist", dist)
                    print(l)
                    raise Exception("here")
                if l != level:
                    toAdd = g.nodes[node][nestedHierarchy[l]]
                else:
                    toAdd = node
                #nested dictionaries
                for l2 in range(topLevel, l+1, -1):
                    desc = g.nodes[node][nestedHierarchy[l2]]
                    if desc not in lookUp:
                        lookUp[desc] = {}
                    lookUp = lookUp[desc]
                #until hit bottom
                if l != topLevel:
                    desc = g.nodes[node][nestedHierarchy[l+1]]
                    if desc not in lookUp:
                        lookUp[desc] = set([])
                    lookUp = lookUp[desc]
                if isinstance(lookUp, dict):
                    lookUp = set([])
                lookUp.add(toAdd)

        else:
            lookUp[node] = [dists, {}]
            subNodes = g.nodes[node]["nestedElements"]
            state = buildDistricting(state, level-1, subNodes)            

    return state

################################################################################

def getMidEdgeCutWeights(node, nbrs, tested, edgeWeights, mergedPop):
    nbrToPopRep = {}
    for nbr in nbrs:
        popRep = edgeWeights[frozenset([node, nbr])]
        if nbr in tested:
            popRep = mergedPop - popRep
        nbrToPopRep[nbr] = popRep
    return nbrToPopRep

################################################################################

def nbrAssignmentsOnCut(node, nodePop, nbrToPopRep, state, level):

    # print(node)
    clockwiseNbrs = state["layeredGraph"]["orientedNeighbors"][level]
    clockwiseNbrs = clockwiseNbrs[node[0]]
    if clockwiseNbrs[0] == clockwiseNbrs[-1]:
        clockwiseNbrs = clockwiseNbrs[:-1]
    clockwiseNbrs = [(cwn, 1) for cwn in clockwiseNbrs 
                     if (cwn, 1) in nbrToPopRep]

    queue = [["A"] + ["B" for n in range(len(nbrToPopRep)-1)]]
    nbrs = clockwiseNbrs

    nbrSplits = []
    queued = set([])

    while len(queue) > 0:
        nbrAssignments = queue.pop(0)
        # print("nbrAssignmentsOnCut queue", nbrAssignments)
        minPops = {"A":0, "B":0}
        for ii in range(len(nbrAssignments)):
            dist = nbrAssignments[ii]
            try:
                nbrPop = nbrToPopRep[nbrs[ii]]
            except:
                print(ii)
                print("node", node)
                print("nbrToPopRep", nbrToPopRep)
                print("clockwiseNbrs", clockwiseNbrs)
                print("clockwiseNbrs[node[0]]", state["layeredGraph"]["orientedNeighbors"][level][node[0]])
                raise Exception("here")
            minPops[dist] += nbrPop
        
        minA = (state["minPop"] <= minPops["A"] + nodePop)
        minB = (state["minPop"] <= minPops["B"] + nodePop)
        maxA = (state["maxPop"] >  minPops["A"])
        maxB = (state["maxPop"] >  minPops["B"])

        if minA and maxA and minB and maxB:
            nAs = {nbrs[ii]:nbrAssignments[ii] for ii in range(len(nbrs))}
            nbrSplits.append(nAs)
        if minPops["A"] < state["maxPop"]:
            try:
                # could be a split district that is a leaf
                firstBInd = nbrAssignments.index("B")
                lastBInd = len(nbrAssignments) - 1 \
                          -nbrAssignments[::-1].index("B")
                if firstBInd == lastBInd:
                    continue
            except:
                continue
            nbrAssignmentsAlt1 = copy.deepcopy(nbrAssignments)
            nbrAssignmentsAlt2 = copy.deepcopy(nbrAssignments)
            nbrAssignmentsAlt1[firstBInd] = "A"
            nbrAssignmentsAlt2[lastBInd] = "A"
            alt1Str = ''.join(nbrAssignmentsAlt1)
            alt2Str = ''.join(nbrAssignmentsAlt2)
            if alt1Str not in queued:
                queue.append(nbrAssignmentsAlt1)
                queued.add(alt1Str)
            if alt2Str not in queued:
                queue.append(nbrAssignmentsAlt2)
                queued.add(alt2Str)

    return nbrSplits

################################################################################

def findCutNodes(edgeWeights, root, mergedPop, mergedGraph, state, level, dists):
    
    queue = [root]
    tested = set([])

    cutNodes = []

    while len(queue) > 0:
        # print("queue", queue)
        node = queue.pop(0)
        tested.add(node)
        nbrs = [nbr for nbr in mergedGraph.neighbors(node) 
                    if frozenset([node, nbr]) in edgeWeights]
        queue += [nbr for nbr in nbrs if nbr not in tested]

        if isinstance(node, tuple):
            nodeDists = getDistricts(state, node[0], level)
        else:
            nodeDists = getDistricts(state, node, level)
        if isinstance(nodeDists, int):
            nodeDists = [nodeDists]
        nodeDists = set(nodeDists)

        if not nodeDists.issubset(dists):
            continue
        # print("continuing...", queue)
        
        nbrToPopRep = getMidEdgeCutWeights(node, nbrs, tested, edgeWeights, 
                                           mergedPop)

        check_huh = True
        for nbr in nbrs:
            if nbrToPopRep[nbr] > state["maxPop"]:
                nbrs = [nbr]
                check_huh = False
        # print("checking...", check_huh, nbrToPopRep)
        
        if check_huh:
            nodePop = mergedGraph.nodes[node]["Population"]
            nbrSplits = nbrAssignmentsOnCut(node, nodePop, nbrToPopRep, state, 
                                            level)
            if len(nbrSplits) > 0:
                # print("foundSplits")
                cutNodes.append([node, nbrSplits, nbrToPopRep]) 
        # raise Exception("here")

    # assert len(cutNodes) <= 1

    return cutNodes

################################################################################

def findNbrNodes(nodeGraph, cut, nbr, state):
    cutNode, cutLevel = cut
    nbrNode, nbrLevel = nbr
    
    nbrNode = nbr
    if isinstance(nbr, tuple):
        nbrNode = nbr[0]

    # if (nbrNode, nbrLevel) in state["districts"]["n2d"]:
    #     state["layeredGraph"]["graphLevels"][]


################################################################################

def findWholeNbrEdgeCounts(nodeGraph, node, nodeLevel, nbrLevel, state):

    if nbrLevel == nodeLevel:
        graph = state["layeredGraph"]["graphLevels"][nodeLevel]
        return graph.nodes[nbr]["downNbrs"][node][1]
    elif nbrLevel > nodeLevel:
        print("WARNING untested; findWholeNbrEdgeCounts; nbrL > nodeL")
        nbrGraph = state["layeredGraph"]["graphLevels"][nbrLevel]
        nodeGraph = state["layeredGraph"]["graphLevels"][nbrLevel]
        nestedHierarchy = state["layeredGraph"]["nestedHierarchy"]
        downNbrs = nbrGraph.nodes[nbr]["downNbrs"][1]
        for l in range(nbrLevel, nodeLevel-1, -1):
            lDesc = nestedHierarchy[l]
            nbrDescAtL = nodeGraph.nodes[node][lDesc]
            downNbrs = downNbrs[nbrDescAtL][1]
        return downNbrs
    else:
        print("WARNING untested; findWholeNbrEdgeCounts; nbrL < nodeL")
        nbrGraph = state["layeredGraph"]["graphLevels"][nbrLevel]
        nodeGraph = state["layeredGraph"]["graphLevels"][nbrLevel]
        nestedHierarchy = state["layeredGraph"]["nestedHierarchy"]
        downNbrs = nodeGraph.nodes[node]["downNbrs"][1]
        for l in range(nodeLevel, nbrLevel, -1):
            lDesc = nestedHierarchy[l]
            nbrDescAtL = nbrGraph.nodes[nbr][lDesc]
            downNbrs = downNbrs[nbrDescAtL][1]
        return {nbr : downNbrs[nbr]}

################################################################################

def getDists(state, node, level):
    nestedHierarchy = state["layeredGraph"]["hierarchy"]
    topLevel = len(nestedHierarchy) - 1
    
    graph = state["layeredGraph"]["graphLevels"][level]
    nodeToDist = state["districts"]["n2d"]

    dists = set([])
    for l in range(topLevel, -1, -1):
        if l > level:
            lDesc = nestedHierarchy[l]
            key = (graph.nodes[node][lDesc], l)
        elif l == level:
            key = (node, l)
        else:
            for nestedNode in graph.nodes[node]["nestedElements"]:
                dists = dists.union(getDists(state, nestedNode, level-1))
            break
        ###
        if key in nodeToDist:
            dists.add(nodeToDist[key])
            break
    return list(dists)

################################################################################

def findDistNbrEdgeCounts(nodeGraph, nodeInfo, subNodesNearNbr, subNbrsNearNode,
                          state, dist, subNodeLevel, subNbrLevel):
    # node, nodeLevel = nodeInfo

    # edgeCounts = {k : [0] for k in subNodesNearNbr.keys()}
    
    # for subNbr in subNbrsNearNode:
    #     dists = getDists(state, subNbr, subNbrLevel)
    #     if dist in dists:
    #         if len(dists) == 1:
    #             nbrGraph = state["layeredGraph"]["graphLevels"][subNbrLevel]
    #             nodeGraph = state["layeredGraph"]["graphLevels"][subNodeLevel]
    #             for nbrNbr in nbrGraph.neighbors(subNbr):
    #                 if 
            
    #         else:
    #             raise Exception("not yet coded")
    
    return 1


################################################################################

def makeNodeGraph(cutNodeInfo, edgeWeights, mergedGraph, state, level):
    cutNode, nbrSplits, nbrToPopRep = cutNodeInfo
    
    coarseGraph = state["layeredGraph"]["graphLevels"][level]
    fineGraph = state["layeredGraph"]["graphLevels"][level-1]

    if isinstance(cutNode, tuple):
        cutNodeStr = cutNode[0]


    subNodes = coarseGraph.nodes[cutNodeStr]["nestedElements"]
    nodeGraph = copy.deepcopy(fineGraph.subgraph(subNodes))

    print("nodeGraph.nodes", nodeGraph.nodes)
    print("cutNodeStr", cutNodeStr)
    print("nbrSplits", nbrSplits)
    print("nbrToPopRep", nbrToPopRep)

    for nbr in nbrSplits[0]:
        print(nbr)
        dist = ""
        if isinstance(nbr, tuple):
            if len(nbr) > 1:
                dist = nbr[1]
            nbrNode = nbr[0] 
        if dist == "":
            nbrConnection = findWholeNbrEdgeCounts(nodeGraph, cutNodeStr,
                                                   nodeLevel, nbrNode, nbrLevel, 
                                                   state)
        else:
            graph = state["layeredGraph"]["graphLevels"][level]
            subNodesNearNbr = graph.nodes[nbrNode]["downNbrs"][cutNodeStr][1]
            subNbrNearNode = graph.nodes[cutNodeStr]["downNbrs"][nbrNode][1]
            nbrConnection = findDistNbrEdgeCounts(nodeGraph, cutNodeStr, 
                                                  subNodesNearNbr, 
                                                  subNodesNearNbr, state, dist,
                                                  level-1, level-1)
        print(nbrConnection)

    return nodeGraph

################################################################################

def getEdgeCut(mergedTree, mergedPop, mergedGraph, state, info, level):
    
    cutEdges, edgeWeights, root = tree.edgeCuts(mergedTree, mergedPop, 
                                                mergedGraph, state, info, 
                                                retRoot = True)
    # print("edgeWeights", edgeWeights)
    # print()
    # print("cut edges and level", cutEdges, level)
    # print("root", root)
    if level > 0 and len(cutEdges) == 0:
        cutNodeInfo = findCutNode(edgeWeights, root, mergedPop, 
                                  mergedGraph, state, info, level)
        if len(cutNodeInfo) > 0:
            nodeGraph = makeNodeGraph(cutNodeInfo, edgeWeights, 
                                      mergedGraph, state, level)
            raise Exception("here1")
            nodeTree = tree.wilson(nodeGraph, info["rng"])
            print(nodeTree.edges)
            nodePop = sum([nodeGraph.nodes[n]["Population"] for n in nodeGraph])
            print(nodeTree.edges)
            cutEdges, level = getEdgeCut(nodeTree, nodePop, nodeGraph, state, 
                                         info, level-1)
            print(cutEdges)
        raise Exception("here")

    return cutEdges, level

################################################################################

def getTopDescsOnEdge(state, nodes, levels):
    level1, level2 = levels
    topLevel = max(state["layeredGraph"]["hierarchy"])
    topDesc = state["layeredGraph"]["hierarchy"][topLevel]

    if level1 == topLevel:
        nodeDesc1 = nodes[0]
    else:
        g = state["layeredGraph"]["graphLevels"][level1]
        nodeDesc1 = g.nodes[nodes[0]][topDesc]

    if level2 == topLevel:
        nodeDesc2 = nodes[1]
    else:
        g = state["layeredGraph"]["graphLevels"][level2]
        nodeDesc2 = g.nodes[nodes[1]][topDesc]

    return nodeDesc1, nodeDesc2

################################################################################

def mergeSharedDistTopNode(state, mergedGraph, persistentEdge):
    nestedHeirarchy = state["layeredGraph"]["hierarchy"]
    topLevel = len(nestedHeirarchy)-1
    topDesc = nestedHeirarchy[topLevel]
    topGraph = state["layeredGraph"]["graphLevels"][topLevel]

    # print(persistentEdge)
    # print(persistentEdge[2])
    # print(persistentEdge[2][0])
    g1 = state["layeredGraph"]["graphLevels"][persistentEdge[2][0]]
    g2 = state["layeredGraph"]["graphLevels"][persistentEdge[2][1]]
    n1 = persistentEdge[0][0]
    n2 = persistentEdge[0][1]

    top1 = g1.nodes[n1][topDesc]
    top2 = g2.nodes[n2][topDesc]

    d1 = getDistricts(state, n1, 0)
    d2 = getDistricts(state, n2, 0)

    # print(mergedGraph.nodes)
    # print("(top1, topLevel, d1)", (top1, topLevel, d1))
    # print("(top2, topLevel, d2)", (top2, topLevel, d2))

    try:
        pop1 = mergedGraph.nodes[(top1, topLevel, d1)]["Population"]
        pop2 = mergedGraph.nodes[(top2, topLevel, d2)]["Population"]
    except:
        print("mergedGraph.nodes", mergedGraph.nodes)
        print("(top1, topLevel, d1)", (top1, topLevel, d1))
        print("(top2, topLevel, d2)", (top2, topLevel, d2))
        for n in mergedGraph.nodes:
            print("node and dict:", n, mergedGraph.nodes[n]["Population"])
        print("topDistGraph", state["layeredGraph"]["topDistGraph"].nodes)
        nestedTrees = state["layeredGraph"]["nestedTrees"]
        # for d in [d1, d2]:
        for d in nestedTrees:
            print("nestedTrees", d, nestedTrees[d])
            print("nestedTree nodes", d, nestedTrees[d]["top"][0].nodes)
            print("nestedTree Graph nodes", d, nestedTrees[d]["top"][1].nodes)
        raise Exception("No pop found")

    node1 = (top1, topLevel, d1)
    node2 = (top2, topLevel, d2)
    if top1 == top2:
        mergedGraph = nx.contracted_nodes(mergedGraph, node1, node2)
        mergedGraph.nodes[node1]["Population"] = pop1 + pop2
    else:
        noE = topGraph.number_of_edges(node1, node2)
        for ii in range(noE):
            mergedGraph.add_edge(node1, node2)   

    nodeMap = {n : n[:2] for n in mergedGraph.nodes}
    mergedGraph = nx.relabel_nodes(mergedGraph, nodeMap)

    return mergedGraph

################################################################################

def buildMergedTopTree(state, info, d1, d2, persistentEdge):
    # print("dists", d1, d2)
    mergedNodes = getTopLevelNodes(state, d1)
    # print("mergedNodes", d1, mergedNodes)
    mergedNodes = mergedNodes.union(getTopLevelNodes(state, d2))
    # print("mergedNodes", d2, getTopLevelNodes(state, d2))

    # print("mergedNodes2", mergedNodes)
    
    topDistGraph = state["layeredGraph"]["topDistGraph"]
    topDistSubGraph = topDistGraph.subgraph(mergedNodes)
    # print("topDistSubGraph.nodes", topDistSubGraph.nodes)
    # print("topDistGraph nodes", topDistGraph.nodes)
    # print("mergedNodes", mergedNodes)
    # print("mergeSharedDistTopNode")
    mergedGraph = mergeSharedDistTopNode(state, topDistSubGraph, persistentEdge)
    # print("finished mergeSharedDistTopNode")
    mergedTree = tree.wilson(mergedGraph, info["rng"])
    mergedPop = sum([topDistGraph.nodes[n]["Population"] 
                     for n in topDistSubGraph])
    return mergedTree, mergedGraph, mergedPop

# def buildMergedTopTree(state, info, d1, d2, persistentEdge):
#     nestedTrees = state["layeredGraph"]["nestedTrees"]
#     mergedGraph = nx.compose(nestedTrees[d1]["top"][1], 
#                              nestedTrees[d2]["top"][1])

#     for d in [d1, d2]:
#         dTopGraph = nestedTrees[d]["top"][1]
#         for n in dTopGraph.nodes:
#             pop = dTopGraph.nodes[n]["Population"]
#             mergedGraph.nodes[n]["Population"] = pop

#     mergedGraph = mergeSharedDistTopNode(state, mergedGraph, d1, d2, 
#                                          persistentEdge)

#     mergedTree = tree.wilson(mergedGraph, info["rng"])
#     mergedPop = sum([mergedGraph.nodes[n]["Population"] 
#                      for n in mergedGraph])
#     return mergedTree, mergedGraph, mergedPop

################################################################################

def getMergedTopTree(state, persistentEdge):
    nestedTrees = state["layeredGraph"]["nestedTrees"]
    topLevel = max(state["layeredGraph"]["hierarchy"])
    
    d1 = getDistricts(state, persistentEdge[0][0], 0)
    d2 = getDistricts(state, persistentEdge[0][1], 0)

    tree1 = nestedTrees[d1]["top"][0]
    tree2 = nestedTrees[d2]["top"][0]

    graph1 = nestedTrees[d1]["top"][1]
    graph2 = nestedTrees[d2]["top"][1]
    
    mergedTree = nx.compose(tree1, tree2)

    # print("mergedTree.nodes", mergedTree.nodes)
    # print("graph1.nodes", graph1.nodes)
    # print("graph2.nodes", graph2.nodes)
    # print("tree1.nodes", tree1.nodes)
    # print("tree1.nodes", tree2.nodes)

    for node in graph1.nodes:
        mergedTree.nodes[node]["Population"] = graph1.nodes[node]["Population"]
    for node in graph2.nodes:
        mergedTree.nodes[node]["Population"] = graph2.nodes[node]["Population"]

    topNodePE1, topNodePE2 = getTopDescsOnEdge(state, persistentEdge[0], 
                                                      persistentEdge[2]) 

    n1 = (topNodePE1, topLevel, d1)
    n2 = (topNodePE2, topLevel, d2)
    # print("top nodes to merge on", n1, n2)
    if topNodePE1 == topNodePE2:
        mergedTree = nx.contracted_nodes(mergedTree, n1, n2)
        mergedTree.nodes[n1]["Population"] += graph2.nodes[n2]["Population"]
    else:
        mergedTree.add_edge(n1, n2)

    nodeMap = {n : n[:2] for n in tree1.nodes}
    nodeMap.update({n : n[:2] for n in tree2.nodes})

    # print("nodeMap", nodeMap)

    mergedTree = nx.relabel_nodes(mergedTree, nodeMap)
    # mergedPop = sum([topDistGraph.nodes[n]["Population"] 
    #                      for n in topDistSubGraph])
    return mergedTree

################################################################################

def countEdgeCuts(state, d1, d2, persistentEdge, info):
    mergedTree = getMergedTopTree(state, persistentEdge)
    # print("old tree 1", d1, state["layeredGraph"]["nestedTrees"][d1]["top"][0].edges)
    # print("old tree 2", d2, state["layeredGraph"]["nestedTrees"][d2]["top"][0].edges)
    # print("mergedTree", mergedTree.nodes)
    # print()
    # print("old mergedTree.nodes", mergedTree.nodes)
    # try:
    mergedPop = sum([mergedTree.nodes[n]["Population"] 
                     for n in mergedTree.nodes])
    # except:
    #     print("EXCEPTING AND EXITING")
    #     nestedTrees = state["layeredGraph"]["nestedTrees"]
    #     print("di nodes", nestedTrees[d1]["top"][1].nodes)
    #     print("di nodes", nestedTrees[d2]["top"][1].nodes)
    #     for n in mergedTree.nodes:        
    #         print(n, mergedTree.nodes[n])

    #     raise Exception("here")
    
    cutEdges, edgeWeights, root = tree.edgeCuts(mergedTree, mergedPop, 
                                                mergedTree, state, 
                                                retRoot = True)
    # print("edgeWeights", edgeWeights)
    # print()
    # print("cut edges", cutEdges)
    # print("root", root)
    # print()
    
    topLevel = max(state["layeredGraph"]["hierarchy"])
    topDesc = state["layeredGraph"]["hierarchy"][topLevel]
    cutNodesInfo = findCutNodes(edgeWeights, root, mergedPop, 
                                mergedTree, state, topLevel, set([d1, d2]))
    nestedTrees = state["layeredGraph"]["nestedTrees"]
    # nestedTrees[d1]
    # print("dists", d1, d2)

    ######
    #rebuild graph based on cutNodes
    rebuildCrossEdges = {}
    for cutNodeInfo in cutNodesInfo:
        cutNode, nbrSplits, nbrToPopRep = cutNodeInfo
        # if(cutNode[0]== "pasquotank"):
        #     print("asking to build tree on pasquotank")
        for d in [d1, d2]:
            # if(cutNode[0] == "pasquotank") and cutNode in nestedTrees[d]:
            #     print("pasquotank before rebuild", d, nestedTrees[d][cutNode][2])
            if cutNode + (d,) in nestedTrees[d]["top"][0].nodes and \
               cutNode not in nestedTrees[d]:
                # if(cutNode[0]== "pasquotank"):
                #     print("building tree on pasquotank", d, cutNode + (d,) in nestedTrees[d]["top"][0].nodes, 
                #           cutNode not in nestedTrees[d])
                #build new cutnode
                # print("building subtree on node", d, cutNode)
                node, level = cutNode
                subGraph = buildSublevelDistGraph(state, node, level, d)
                subTree = tree.wilson(subGraph, info["rng"])
                nestedTrees[d][cutNode] = [subTree, subGraph, set([]), []]
                if d in rebuildCrossEdges:
                    rebuildCrossEdges[d].add(cutNode + (d,))
                else:
                    rebuildCrossEdges[d] = set([cutNode + (d,)])

    fineGraph = state["layeredGraph"]["graphLevels"][0]
    for d in rebuildCrossEdges:
        rebuildNodeEdges = rebuildCrossEdges[d]
        distTree = nestedTrees[d]["top"][0]
        # print(distTree.edges)
        # print(rebuildEdges)

        for node in rebuildNodeEdges:
            for nbr in distTree.neighbors(node):
                nbrKey = nbr[:2]
                # print("node to rebuild edges on", node, "and nbr", nbr)
                if nbrKey in nestedTrees[d]:
                    existingEdges = nestedTrees[d][nbrKey][2]
                    edge = False
                    setEdge = True
                    for e in existingEdges:
                        n1, n2 = list(e)
                        top1, top2 = n1[0], n2[0]
                        if n1[1] == 0:
                            top1 = fineGraph.nodes[n1[0]]["County"]
                        if n2[1] == 0:
                            top2 = fineGraph.nodes[n2[0]]["County"]
                        # print("e in existingEdges", node[:2], e)
                        tops = set([top1, top2])
                        if node[:2] in e:
                            # print("found edge", e, existingEdges)
                            oldEdge = e
                            nbr = list(oldEdge - set([node[:2]]))[0]
                            break
                        elif node[0] in tops and nbr[0] in tops and n1[1] == 0\
                             and n2[1] == 0:
                             setEdge = False

                    if not setEdge:
                        continue
                    # print(node, nbr)
                    edge = setCrossEdge(state, node[:2], nbr[:2], set([d]), info, 
                                        cut12 = {1: True, 2: True})
                    # print("new edge", edge)
                    try:
                        nestedTrees[d][nbrKey][2].remove(oldEdge)
                    except:
                        # print("node", node)
                        # print("existingEdges", existingEdges)
                        # print("adding", edge, "all", nestedTrees[d][nbrKey][2], "dist", d)
                        # print("adding", edge, "removing", oldEdge, "all", nestedTrees[d][nbrKey][2])
                        # raise Exception("here")
                        pass
                    nestedTrees[d][node[:2]][2].add(edge)
                    nestedTrees[d][nbrKey][2].add(edge)
                else:
                    edge = setCrossEdge(state, node[:2], nbrKey, set([d]), info, 
                                        cut12 = {1: True, 2: False})
                    nestedTrees[d][node[:2]][2].add(edge)
    #########
    #getMergedBotTree(state, d1, d2, cutNodesInfo)

    nestedTrees = state["layeredGraph"]["nestedTrees"]
    fineGraph = state["layeredGraph"]["graphLevels"][0]
    for d in nestedTrees:
        for cntyLevel in nestedTrees[d]:
            if cntyLevel == "top":
                continue
            # if(cntyLevel[0]== "pasquotank") and cntyLevel in nestedTrees[d]:
            #     print("pasquotank before adding rebuild edges", nestedTrees[d][cntyLevel][2])
            outEdges = nestedTrees[d][cntyLevel][2]
            outCntys = set([])
            for oE in outEdges:
                n1, n2 = list(oE)
                top1, top2 = n1[0], n2[0]
                if n1[1] == 0:
                    top1 = fineGraph.nodes[n1[0]]["County"]
                if n2[1] == 0:
                    top2 = fineGraph.nodes[n2[0]]["County"]
                outCnty = list(set([top1, top2]) - set([cntyLevel[0]]))[0]
                # if(cntyLevel[0]== "pasquotank") and cntyLevel in nestedTrees[d]:
                #     print("outCnty", outCnty, oE)
                if outCnty in outCntys:
                    print(nestedTrees)
                    raise Exception("here3", d, cntyLevel, step, outCnty)

    possibleCuts = set([])
    for cutNodeInfo in cutNodesInfo:
        cutNode, nbrSplits, nbrToPopRep = cutNodeInfo
        cutTree = nx.Graph()
        for d in [d1, d2]:
            if cutNode in nestedTrees[d]:
                cutTree = nx.compose(cutTree, nestedTrees[d][cutNode][0])
                for node in nestedTrees[d][cutNode][0].nodes:
                    pop = nestedTrees[d][cutNode][1].nodes[node]["Population"]
                    cutTree.nodes[node]["Population"] = pop

        # if(cutNode[0]== "pasquotank"):
        #     print("cutTree", cutTree.edges)
        # print()
        # print(cutNode)
        # print("nbrToPopRep", nbrToPopRep)
        # print(nestedTrees[5][cutNode][1].nodes['1691'])
        # print(persistentEdge[0], cutTree.nodes)
        pe1, pe2 = persistentEdge[0]
        fineGraph = state["layeredGraph"]["graphLevels"][0]
        if pe1 in cutTree.nodes and pe2 in cutTree.nodes:
           cutTree.add_edge(pe1, pe2)
        elif pe1 in cutTree.nodes:
            topName = fineGraph.nodes[pe2][topDesc]
            topNode = (topName, topLevel)
            cutTree.add_edge(pe1, topNode)
            cutTree.nodes[topNode]["Population"] = nbrToPopRep[topNode]
        elif pe2 in cutTree.nodes:
            topName = fineGraph.nodes[pe1][topDesc]
            topNode = (topName, topLevel)
            cutTree.add_edge(pe2, topNode)
            cutTree.nodes[topNode]["Population"] = nbrToPopRep[topNode]

        # if(cutNode[0]== "pasquotank"):
        #     print("cutTree w persistentEdge", cutTree.edges)
        #     print("persistentEdge", persistentEdge)
        #     for d in [d1, d2]:
        #         if cutNode not in nestedTrees[d]:
        #             continue
        #         print("outEdges", d, nestedTrees[d][cutNode][2])
        
        edgeMap = {}
        for d in [d1, d2]:
            if cutNode not in nestedTrees[d]:
                continue
            outEdges = nestedTrees[d][cutNode][2]
            # outEdgesTop = []
            # print("dist and cutnode in loop", d, cutNode, outEdges)
            for oE in list(outEdges):
                n1, n2 = list(oE)
                # if(cutNode[0]== "pasquotank"):
                #     print("outedgeoE, n1, n2", oE, n1, n2)
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
                # if(cutNode[0]== "pasquotank"):
                #     print("outEdge", d, inNode, outNode)

                # print("out node in", outNode, outNode not in nbrToPopRep)
                if outNode not in nbrToPopRep:
                    continue

                # print(cutNode, inNode, outNode)

                edgeMap[frozenset([outNode, inNode])] = oE
                outEdgePop = nbrToPopRep[outNode]
                # print(outEdgePop)
                cutTree.add_edge(outNode, inNode)
                cutTree.nodes[outNode]["Population"] = outEdgePop
                
            # print("nestedTree out edges", d, nestedTrees[d][cutNode][2])
                # for nbr in nbrToPopRep:
                #     if nbr in nestedTrees[d][cutNode][2]:


        # print("edges", cutTree.edges)
        # pop = 0
        # for n in cutTree.nodes:
        #     pop += cutTree.nodes[n]["Population"]
        # print(mergedPop, pop)
        # print("outEdges", outEdges)

        try:
            subNodeCutInfo = tree.edgeCuts(cutTree, mergedPop, cutTree, state, 
                                           retRoot = True)
        except:
            print("cutTree that failed", cutTree.edges, cutNodeInfo[0], [d1, d2])
            raise Exception("failed")
        newCuts = set([])

        for cut in subNodeCutInfo[0]:
            if cut in edgeMap:
                newCuts.add(edgeMap[cut])
            else:
                newCuts.add(cut)

        possibleCuts.update(newCuts)
    if len(possibleCuts) == 0 and len(cutEdges) > 0:
        return cutEdges
 
    return possibleCuts#noEdgesToCut

################################################################################

def countEdgeCutsDB(state, d1, d2, persistentEdge, info):
    mergedTree = getMergedTopTree(state, persistentEdge)
    # print("old tree 1", d1, state["layeredGraph"]["nestedTrees"][d1]["top"][0].edges)
    # print("old tree 2", d2, state["layeredGraph"]["nestedTrees"][d2]["top"][0].edges)
    # print("mergedTree", mergedTree.nodes)
    # print()
    # print("old mergedTree.nodes", mergedTree.nodes)
    # try:
    mergedPop = sum([mergedTree.nodes[n]["Population"] 
                     for n in mergedTree.nodes])
    # except:
    #     print("EXCEPTING AND EXITING")
    #     nestedTrees = state["layeredGraph"]["nestedTrees"]
    #     print("di nodes", nestedTrees[d1]["top"][1].nodes)
    #     print("di nodes", nestedTrees[d2]["top"][1].nodes)
    #     for n in mergedTree.nodes:        
    #         print(n, mergedTree.nodes[n])

    #     raise Exception("here")
    
    cutEdges, edgeWeights, root = tree.edgeCuts(mergedTree, mergedPop, 
                                                mergedTree, state, 
                                                retRoot = True)


    print("edgeWeights", edgeWeights)
    print()
    print("cut edges", cutEdges)
    print("root", root)
    print()
    
    topLevel = max(state["layeredGraph"]["hierarchy"])
    topDesc = state["layeredGraph"]["hierarchy"][topLevel]
    cutNodesInfo = findCutNodes(edgeWeights, root, mergedPop, 
                                mergedTree, state, topLevel, set([d1, d2]))
    nestedTrees = state["layeredGraph"]["nestedTrees"]
    # nestedTrees[d1]
    # print("dists", d1, d2)

    ######
    #rebuild graph based on cutNodes
    rebuildCrossEdges = {}
    for cutNodeInfo in cutNodesInfo:
        cutNode, nbrSplits, nbrToPopRep = cutNodeInfo
        for d in [d1, d2]:
            if cutNode + (d,) in nestedTrees[d]["top"][0].nodes and \
               cutNode not in nestedTrees[d]:
                #build new cutnode
                # print("building subtree on node", d, cutNode)
                node, level = cutNode
                subGraph = buildSublevelDistGraph(state, node, level, d)
                subTree = tree.wilson(subGraph, info["rng"])
                nestedTrees[d][cutNode] = [subTree, subGraph, set([]), []]
                if d in rebuildCrossEdges:
                    rebuildCrossEdges[d].add(cutNode + (d,))
                else:
                    rebuildCrossEdges[d] = set([cutNode + (d,)])

    fineGraph = state["layeredGraph"]["graphLevels"][0]
    for d in rebuildCrossEdges:
        rebuildNodeEdges = rebuildCrossEdges[d]
        distTree = nestedTrees[d]["top"][0]
        # print(distTree.edges)
        # print(rebuildEdges)

        for node in rebuildNodeEdges:
            for nbr in distTree.neighbors(node):
                nbrKey = nbr[:2]
                # print("node to rebuild edges on", node, "and nbr", nbr)
                if nbrKey in nestedTrees[d]:
                    existingEdges = nestedTrees[d][nbrKey][2]
                    edge = False
                    setEdge = True
                    for e in existingEdges:
                        n1, n2 = list(e)
                        top1, top2 = n1[0], n2[0]
                        if n1[1] == 0:
                            top1 = fineGraph.nodes[n1[0]]["County"]
                        if n2[1] == 0:
                            top2 = fineGraph.nodes[n2[0]]["County"]
                        # print("e in existingEdges", node[:2], e)
                        tops = set([top1, top2])
                        if node[:2] in e:
                            # print("found edge", e, existingEdges)
                            oldEdge = e
                            nbr = list(oldEdge - set([node[:2]]))[0]
                            break
                        elif node[0] in tops and nbr[0] in tops and n1[1] == 0\
                             and n2[1] == 0:
                             setEdge = False

                    if not setEdge:
                        continue
                    # print(node, nbr)
                    edge = setCrossEdge(state, node[:2], nbr[:2], set([d]), info, 
                                        cut12 = {1: True, 2: True})
                    # print("new edge", edge)
                    try:
                        nestedTrees[d][nbrKey][2].remove(oldEdge)
                    except:
                        # print("node", node)
                        # print("existingEdges", existingEdges)
                        # print("adding", edge, "all", nestedTrees[d][nbrKey][2], "dist", d)
                        # print("adding", edge, "removing", oldEdge, "all", nestedTrees[d][nbrKey][2])
                        # raise Exception("here")
                        pass
                    nestedTrees[d][node[:2]][2].add(edge)
                    nestedTrees[d][nbrKey][2].add(edge)
                else:
                    edge = setCrossEdge(state, node[:2], nbrKey, set([d]), info, 
                                        cut12 = {1: True, 2: False})
                    nestedTrees[d][node[:2]][2].add(edge)
    #########
    #getMergedBotTree(state, d1, d2, cutNodesInfo)

    nestedTrees = state["layeredGraph"]["nestedTrees"]
    fineGraph = state["layeredGraph"]["graphLevels"][0]
    for d in nestedTrees:
        for cntyLevel in nestedTrees[d]:
            if cntyLevel == "top":
                continue
            outEdges = nestedTrees[d][cntyLevel][2]
            outCntys = set([])
            for oE in outEdges:
                n1, n2 = list(oE)
                top1, top2 = n1[0], n2[0]
                if n1[1] == 0:
                    top1 = fineGraph.nodes[n1[0]]["County"]
                if n2[1] == 0:
                    top2 = fineGraph.nodes[n2[0]]["County"]
                outCnty = list(set([top1, top2]) - set([cntyLevel[0]]))[0]
                if outCnty in outCntys:
                    print(nestedTrees)
                    raise Exception("here3", d, cntyLevel, step, outCnty)

    possibleCuts = set([])
    for cutNodeInfo in cutNodesInfo:
        cutNode, nbrSplits, nbrToPopRep = cutNodeInfo
        cutTree = nx.Graph()
        for d in [d1, d2]:
            if cutNode in nestedTrees[d]:
                print("nestedTrees[d]", nestedTrees[d][cutNode][0].edges)
                cutTree = nx.compose(cutTree, nestedTrees[d][cutNode][0])
                for node in nestedTrees[d][cutNode][0].nodes:
                    pop = nestedTrees[d][cutNode][1].nodes[node]["Population"]
                    cutTree.nodes[node]["Population"] = pop
        # print()
        print(cutNode)
        print("nbrToPopRep", nbrToPopRep)
        print("cutTree", cutTree.edges)
        # print(persistentEdge[0], cutTree.nodes)
        pe1, pe2 = persistentEdge[0]
        fineGraph = state["layeredGraph"]["graphLevels"][0]
        if pe1 in cutTree.nodes and pe2 in cutTree.nodes:
           cutTree.add_edge(pe1, pe2)
        elif pe1 in cutTree.nodes:
            topName = fineGraph.nodes[pe2][topDesc]
            topNode = (topName, topLevel)
            cutTree.add_edge(pe1, topNode)
            cutTree.nodes[topNode]["Population"] = nbrToPopRep[topNode]
        elif pe2 in cutTree.nodes:
            topName = fineGraph.nodes[pe1][topDesc]
            topNode = (topName, topLevel)
            cutTree.add_edge(pe2, topNode)
            cutTree.nodes[topNode]["Population"] = nbrToPopRep[topNode]

        print("cutTree w persistentEdge", cutTree.edges)
        
        edgeMap = {}
        for d in [d1, d2]:
            if cutNode not in nestedTrees[d]:
                continue
            outEdges = nestedTrees[d][cutNode][2]
            # outEdgesTop = []
            # print("dist and cutnode in loop", d, cutNode, outEdges)
            for oE in list(outEdges):
                n1, n2 = list(oE)
                print("outedgeoE, n1, n2", oE, n1, n2)
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
                print("adding outEdge", d, inNode, outNode)

                # print("out node in", outNode, outNode not in nbrToPopRep)
                if outNode not in nbrToPopRep:
                    continue

                # print(cutNode, inNode, outNode)

                edgeMap[frozenset([outNode, inNode])] = oE
                outEdgePop = nbrToPopRep[outNode]
                # print(outEdgePop)
                cutTree.add_edge(outNode, inNode)
                cutTree.nodes[outNode]["Population"] = outEdgePop
            
            # print("nestedTree out edges", d, nestedTrees[d][cutNode][2])
                # for nbr in nbrToPopRep:
                #     if nbr in nestedTrees[d][cutNode][2]:

        print("cutTree w outside edges", cutTree.edges)
        # print("edges", cutTree.edges)
        # pop = 0
        # for n in cutTree.nodes:
        #     pop += cutTree.nodes[n]["Population"]
        # print(mergedPop, pop)
        # print("outEdges", outEdges)

        subNodeCutInfo = tree.edgeCuts(cutTree, mergedPop, cutTree, state, 
                                       retRoot = True)
        print(subNodeCutInfo)
        newCuts = set([])

        for cut in subNodeCutInfo[0]:
            if cut in edgeMap:
                newCuts.add(edgeMap[cut])
            else:
                newCuts.add(cut)

        possibleCuts.update(newCuts)
    if len(possibleCuts) == 0 and len(cutEdges) > 0:
        return cutEdges
 
    return possibleCuts#noEdgesToCut