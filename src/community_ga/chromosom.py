import math
import time
from itertools import product

import numpy as np
from networkx.algorithms.community.community_utils import is_partition

try:
    from . import main
except ImportError:  # when main is executed as a script
    import __main__ as main


def __modularity(communities, weight='weight') -> (list, int):
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(main.G, communities):
        print("not a partition")
        return

    multigraph = main.G.is_multigraph()
    directed = main.G.is_directed()
    m = main.G.size(weight=weight)
    if directed:
        out_degree = dict(main.G.out_degree(weight=weight))
        in_degree = dict(main.G.in_degree(weight=weight))
        norm = 1 / m
    else:
        out_degree = dict(main.G.degree(weight=weight))
        in_degree = out_degree
        norm = 1 / (2 * m)

    def val(u, v):
        try:
            if multigraph:
                w = sum(d.get(weight, 1) for k, d in main.G[u][v].items())
            else:
                w = main.G[u][v].get(weight, 1)
        except KeyError:
            w = 0
        # Double count self-loops if the graph is undirected.
        if u == v and not directed:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Qs = [0] * len(communities)
    Q = 0
    tmp = 0
    for c in range(len(communities)):
        tmp += len(list(product(communities[c], repeat=2)))
        for u, v in product(communities[c], repeat=2):
            Qs[c] += norm * val(u, v)
        Q += Qs[c]
    print("product len = " + str(tmp))
    return (Qs, Q)


class Chromosom():
    def __init__(self, V=None, empty=False):
        self.Q = 0
        self.Qs = {}
        self.V = {}
        self.maxCMID = 0
        self.m_chr = [False] * len(main.mst.edges)
        self.cmV = dict.fromkeys(list(main.mst.nodes), float("inf"))
        if (V != None):
            self.V = V
            self.setChr()
        elif (empty == False):
            self.randomChromosom2()
            self.findCommunities()
            self.updateQ_Qs()

    def updateQ_Qs(self) -> (float, float):
        qtime = time.time()
        (self.Qs, self.Q) = self.myModularity()
        qtime = time.time() - qtime
        mtime = time.time()

        self.Qs = {k: v for k, v in sorted(
            self.Qs.items(), key=lambda item: item[1], reverse=True)}
        # (self.V, self.Qs) = mergeSort(self.V.copy(), self.Qs.copy())*
        mtime = time.time() - mtime
        return (qtime, mtime)

    def BFS(self, startNode, fatherNode, m, T) -> (dict, list):
        brokenEdges = []
        searchList = [startNode]
        searchListFs = [fatherNode]
        t = 0
        while (len(searchList) != 0):
            n = searchList.pop(0)
            nN = searchListFs.pop(0)
            if (t < T):
                m[n] = True
                t = t + 1

                mNeighbors = list(main.mst.neighbors(n))
                for neighbor in mNeighbors:
                    if (not (m[neighbor])):
                        searchList.extend([neighbor])
                        searchListFs.extend([n])
                    else:
                        if (self.m_chr[main.mst.edges[n, neighbor]["index"]] == True):
                            brokenEdges.append([n, neighbor])
            else:
                self.m_chr[main.mst.edges[n, nN]["index"]] = True
                brokenEdges.append([n, neighbor])
        if (t < T):
            if ((len(brokenEdges)-1) < 0):
                if (startNode != fatherNode):
                    self.m_chr[main.mst.edges[startNode,
                                              fatherNode]["index"]] = False
            else:
                rnd = main.mRand.randint(0, (len(brokenEdges)-1))
                self.m_chr[main.mst.edges[brokenEdges[rnd][0],
                                          brokenEdges[rnd][1]]["index"]] = False
        return (m, brokenEdges)

    def randomChromosom2(self):
        m = dict.fromkeys(list(main.mst.nodes), False)
        rnd = main.mRand.randint(0, (len(main.edgeList) - 1))
        borderEdges = [[main.edgeList[rnd][0], main.edgeList[rnd][1]]]
        T = int(np.ceil((np.sqrt(len(main.mst.nodes)))))

        while (len(borderEdges) != 0):
            edge = borderEdges.pop()
            if (not (m[edge[1]]) and not (m[edge[0]])):
                res = self.BFS(edge[0], edge[1], m, T)
                m = res[0]
                borderEdges.extend(res[1])
            elif (not (m[edge[1]]) and m[edge[0]]):
                res = self.BFS(edge[1], edge[0], m, T)
                m = res[0]
                borderEdges.extend(res[1])
            elif (m[edge[1]] and not (m[edge[0]])):
                res = self.BFS(edge[0], edge[1], m, T)
                m = res[0]
                borderEdges.extend(res[1])

    def randomChromosom(self):
        seq = list(main.mst.nodes)
        seq = main.mRand.sample(seq, len(seq))
        fixed = [False] * len(main.mst.edges)
        checkedNode = dict.fromkeys(seq, False)
        T = int(np.ceil((np.sqrt(len(main.mst.nodes)))))
        for n in seq:
            if checkedNode[n]:
                continue
            checkedNode[n] = True
            cList = []
            bList = []
            expanded = [False] * len(main.mst.edges)
            node = list(main.mst.neighbors(n))[0]
            if self.m_chr[main.mst.edges[n, node]["index"]] is False:
                cList.append(main.mst.edges[n, node]["index"])
            elif self.m_chr[main.mst.edges[n, node]["index"]] is True:
                bList.append(main.mst.edges[n, node]["index"])
            expanded[main.mst.edges[n, node]["index"]] = True
            t = 1
            for cItem in cList:
                t = t + 1
                if t > T and fixed[cItem] is False:
                    self.m_chr[main.mst.edges[main.edgeList[cItem]
                                              [0], main.edgeList[cItem][1]]["index"]] = True
                else:
                    fixed[cItem] = True
                    endNodes = main.edgeList[cItem]
                    checkedNode[endNodes[0]] = True
                    checkedNode[endNodes[1]] = True
                    for neighbor in main.mst.neighbors(endNodes[0]):
                        if expanded[main.mst.edges[endNodes[0], neighbor]["index"]] is False and self.m_chr[main.mst.edges[endNodes[0], neighbor]["index"]] is False:
                            cList.append(
                                main.mst.edges[endNodes[0], neighbor]["index"])
                        elif self.m_chr[main.mst.edges[endNodes[0], neighbor]["index"]] is True:
                            bList.append(
                                main.mst.edges[endNodes[0], neighbor]["index"])
                        expanded[main.mst.edges[endNodes[0], neighbor]
                                 ["index"]] = True
                    for neighbor in main.mst.neighbors(endNodes[1]):
                        if expanded[main.mst.edges[endNodes[1], neighbor]["index"]] is False and self.m_chr[main.mst.edges[endNodes[1], neighbor]["index"]] is False:
                            cList.append(
                                main.mst.edges[endNodes[1], neighbor]["index"])
                        elif self.m_chr[main.mst.edges[endNodes[1], neighbor]["index"]] is True:
                            bList.append(
                                main.mst.edges[endNodes[1], neighbor]["index"])
                        expanded[main.mst.edges[endNodes[1], neighbor]
                                 ["index"]] = True
            if t < T:
                ind = main.mRand.randint(0, len(bList) - 1)
                endNodes = main.edgeList[bList[ind]]
                fixed[bList[ind]] = True
                checkedNode[endNodes[0]] = True
                checkedNode[endNodes[1]] = True
                self.m_chr[bList[ind]] = False

    def setChr(self):
        for V in list(self.V.keys):
            for node in list(self.V[V]):
                for node2 in main.mst.neighbors(node):
                    if (node2 in list(self.V[V])):
                        self.m_chr[main.mst.edges[node, node2]
                                   ["index"]] = False
                        # self.mst.edges[node, node2]["gene"] = 0
                    else:
                        self.m_chr[main.mst.edges[node, node2]["index"]] = True
                        # self.mst.edges[node, node2]["gene"] = 1

    def myModularity(self) -> (dict, float):
        def modCM(c) -> float:
            q = 0
            Ec = 0
            Et = 0
            for node in self.V[c]:
                for nNode in list(main.G.neighbors(node)):
                    if ("weight" in main.G.edges[node, nNode]):
                        Et += main.G.edges[node, nNode]["weight"]
                        if (self.cmV[node] == self.cmV[nNode]):
                            Ec += main.G.edges[node, nNode]["weight"]
                    else:
                        Et += 1
                        if (self.cmV[node] == self.cmV[nNode]):
                            Ec += 1
            if Et > 0:
                q = (Ec / main.m) - ((Et / main.m) ** 2)
            return q

        Q = 0
        if (len(self.Qs) == len(self.V)):
            Qs = self.Qs
        else:
            Qs = dict.fromkeys(list(self.V.keys()), None)
        for c in list(self.V.keys()):
            if (Qs[c] == None):
                Qs[c] = modCM(c)
            Q = Q + Qs[c]
        return (Qs, Q)

    def connComp(self, cm, sNode, _V) -> (list, list, dict):
        # cm = list(self.V[cmId])
        cm1 = [sNode]
        nonConnected = []
        i = 0
        while (i < len(cm1)):
            if _V[cm1[i]] != False:
                i += 1
                continue
            _V[cm1[i]] = True
            neighbrs = set(main.mst.neighbors(cm1[i])).intersection(set(cm))
            for nodeT in neighbrs:
                if self.m_chr[main.mst.edges[cm1[i], nodeT]["index"]] == False:
                    if _V[nodeT] == False:
                        cm1.append(nodeT)
                else:
                    nonConnected.append(nodeT)
            i += 1
        return (cm1, nonConnected, _V)

    def updateCommunities(self, edgeS: int, edgeT: int):
        cmMax = max(self.cmV[edgeS], self.cmV[edgeT])
        cmMin = min(self.cmV[edgeS], self.cmV[edgeT])
        if (self.m_chr[main.mst.edges[edgeS, edgeT]["index"]] == False):
            self.V[cmMin] = self.V[cmMin].union(self.V[cmMax])
            for nodeInV in self.V[cmMax]:
                self.cmV[nodeInV] = cmMin
            del self.V[cmMax]
            self.Q = self.Q-self.Qs[cmMax]
            self.Q = self.Q-self.Qs[cmMin]
            del self.Qs[cmMax]
            self.Qs[cmMin] = self.modCM(cmMin)
            self.Q = self.Q+self.Qs[cmMin]
        else:
            cm = list(self.V[cmMin])
            _V = dict.fromkeys(cm, False)
            (cm1, nonConnected, _V) = self.connComp(
                list(self.V[cmMin]), edgeS, _V)
            (cm2, nonConnected, _V) = self.connComp(
                list(self.V[cmMin]), nonConnected[0], _V)

            for node in cm2:
                self.cmV[node] = self.maxCMID

            self.maxCMID = self.maxCMID+1

            if (len(cm) != (len(cm1) + len(cm2))):
                print("ERROR")

            self.V[cmMin] = set(cm1)
            self.Q = self.Q-self.Qs[cmMin]
            self.V[self.maxCMID-1] = set(cm2)
            self.Qs[cmMin] = self.modCM(cmMin)
            self.Q = self.Q+self.Qs[cmMin]
            self.Qs[self.maxCMID - 1] = self.modCM(self.maxCMID - 1)
            self.Q = self.Q + self.Qs[self.maxCMID - 1]

    def findCommunities(self):
        self.V.clear()
        _V = dict.fromkeys(list(main.mst.nodes.keys()), False)
        self.maxCMID = 0
        for nodeS in main.mst.nodes:
            if _V[nodeS]:
                continue
            i = 0
            _V[nodeS] = True
            communMem = list()
            communMem.append(nodeS)
            self.cmV[nodeS] = self.maxCMID
            while (i < len(communMem)):
                for nodeT in main.mst.neighbors(communMem[i]):
                    if (self.m_chr[main.mst.edges[communMem[i], nodeT]["index"]] == False and _V[nodeT] == False):
                        _V[nodeT] = True
                        communMem.append(nodeT)
                        self.cmV[nodeT] = self.maxCMID
                i += 1
            self.V[self.maxCMID] = set(communMem)
            self.maxCMID += 1

    def mutuate(self, edgeS: int, edgeT: int):
        # don't forget to run updateCommunities afterwards
        if (self.m_chr[main.mst.edges[edgeS, edgeT]["index"]] == False):
            self.m_chr[main.mst.edges[edgeS, edgeT]["index"]] = True
        else:
            self.m_chr[main.mst.edges[edgeS, edgeT]["index"]] = False
        self.updateCommunities(edgeS=edgeS, edgeT=edgeT)

    def modCM(self, c) -> float:
        q = 0
        Ec = 0
        Et = 0
        for node in self.V[c]:
            for nNode in list(main.G.neighbors(node)):
                if ("weight" in main.G.edges[node, nNode]):
                    Et += main.G.edges[node, nNode]["weight"]
                    if (self.cmV[node] == self.cmV[nNode]):
                        Ec += main.G.edges[node, nNode]["weight"]
                else:
                    Et += 1
                    if (self.cmV[node] == self.cmV[nNode]):
                        Ec += 1
        if Et > 0:
            q = (Ec / main.m) - ((Et / main.m) ** 2)
        return q

    def connCom(self, chrPar, cmInd, _V, breakList) -> (list, list, list):
        cm = chrPar.V[cmInd]
        cms = []
        searchList = []
        for sNode in cm:
            if _V[sNode] == False:
                searchList.append(sNode)
            cm1 = []
            while (len(searchList) != 0):
                n = searchList.pop(0)
                cm1.append(n)
                _V[n] = True
                nghbrs = main.mst.neighbors(n)
                for tNode in nghbrs:
                    if (_V[tNode] == False):
                        if (chrPar.m_chr[main.mst.edges[n, tNode]["index"]] == False):
                            if (self.m_chr[main.mst.edges[n, tNode]["index"]] == False):
                                searchList.append(tNode)
                            else:
                                breakList.append(
                                    main.mst.edges[n, tNode]["index"])
                                self.m_chr[main.mst.edges[n, tNode]
                                           ["index"]] = True
                        else:
                            breakList.append(main.mst.edges[n, tNode]["index"])
                            self.m_chr[main.mst.edges[n, tNode]
                                       ["index"]] = True
            if (cm1 != []):
                cms.append(cm1)
        return (cms, _V, breakList)

    def crossOver(self, chr1, chr2) -> list:
        Inds = list(range(len(chr1.V) + len(chr2.V)))
        Qs = list(chr1.Qs.values()) + list(chr2.Qs.values())
        (Inds, Qs) = mergeSort(V=Inds, Qs=Qs)
        r = len(main.mst.nodes) - 1
        c = 0
        self.Q = 0
        self.cmV = dict.fromkeys(list(main.mst.nodes), float("inf"))
        _V = dict.fromkeys(main.mst.nodes, False)
        self.V = {}
        self.Qs = {}
        breakList = []
        self.maxCMID = 0
        while r >= 0:
            tmp = 0
            ind = Inds[c]
            if ind < len(chr1.V):
                (cms, _V, breakList) = self.connCom(
                    chr1, list(chr1.Qs.keys())[ind], _V, breakList)
                for cm in cms:
                    self.V[self.maxCMID] = set(cm)
                    for node in cm:
                        self.cmV[node] = self.maxCMID
                        tmp += 1
                    if (len(cm) == len(chr1.V[list(chr1.Qs.keys())[ind]])):
                        self.Qs[self.maxCMID] = Qs[c]
                        self.Q = self.Q+self.Qs[self.maxCMID]
                        self.maxCMID += 1
                    elif (len(cm) > 0):
                        self.Qs[self.maxCMID] = self.modCM(self.maxCMID)
                        self.Q = self.Q + self.Qs[self.maxCMID]
                        self.maxCMID += 1
                    else:
                        pass
            else:
                ind = ind - len(chr1.V)
                (cms, _V, breakList) = self.connCom(
                    chr2, list(chr2.Qs.keys())[ind], _V, breakList)
                for cm in cms:
                    self.V[self.maxCMID] = set(cm)
                    for node in cm:
                        self.cmV[node] = self.maxCMID
                        tmp += 1
                    if (len(cm) == len(chr2.V[list(chr2.Qs.keys())[ind]])):
                        self.Qs[self.maxCMID] = Qs[c]
                        self.Q = self.Q + self.Qs[self.maxCMID]
                        self.maxCMID += 1
                    elif (len(cm) > 0):
                        self.Qs[self.maxCMID] = self.modCM(self.maxCMID)
                        self.Q = self.Q + self.Qs[self.maxCMID]
                        self.maxCMID += 1
                    else:
                        pass
            r = r - tmp
            c = c + 1
        self.Qs = {k: v for k, v in sorted(
            self.Qs.items(), key=lambda item: item[1], reverse=True)}
        return breakList


def mergeSort(V, Qs) -> (dict, dict):
    def merge(V1, Qs1, V2, Qs2) -> (list, list):
        Qs = [0] * (len(V1) + len(V2))
        V = [0] * (len(V1) + len(V2))
        i = 0
        j = 0
        q = 0
        while (i < len(V1) and j < len(V2)):
            if (Qs1[i] > Qs2[j]):
                Qs[q] = Qs1[i]
                V[q] = V1[i]
                q = q + 1
                i = i + 1
            else:
                Qs[q] = Qs2[j]
                V[q] = V2[j]
                q = q + 1
                j = j + 1

        if (i < len(Qs1)):
            for i in range(i, len(Qs1)):
                Qs[q] = Qs1[i]
                V[q] = V1[i]
                q = q + 1
        if (j < len(Qs2)):
            for j in range(j, len(Qs2)):
                Qs[q] = Qs2[j]
                V[q] = V2[j]
                q = q + 1
        return (V, Qs)
    if (len(Qs) == 1):
        return (V, Qs)
    elif (len(Qs) == 2):
        if (Qs[0] < Qs[1]):
            t = Qs[0]
            Qs[0] = Qs[1]
            Qs[1] = t
            t = V[0]
            V[0] = V[1]
            V[1] = t
        return (V, Qs)
    else:
        mid = math.floor(len(Qs) / 2)
        (V1, Qs1) = mergeSort(V[0:mid], Qs[0: mid])
        (V2, Qs2) = mergeSort(V[mid:len(V)], Qs[mid:len(Qs)])
        (V, Qs) = merge(V1.copy(), Qs1.copy(), V2.copy(), Qs2.copy())
        return (V, Qs)
