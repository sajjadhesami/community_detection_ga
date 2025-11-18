from concurrent.futures import ThreadPoolExecutor
import math
import os
import pathlib
import random
import time
from math import log
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import netwulf
import networkx as nx
import numpy as np
from webweb import Web

try:
    from .chromosom import Chromosom
except ImportError:  # allow running as plain script
    from chromosom import Chromosom


def jaccard(G, nodeID1, nodeID2) -> float:
    return len(set(G.neighbors(nodeID1)).intersection(set(G.neighbors(nodeID2)))) / len(
        set(G.neighbors(nodeID1)).union(G.neighbors(nodeID2)))


def CNDP(G: nx.Graph, Beta=1.76) -> dict:
    r = dict()
    ccs = dict()
    cc_avg = 0
    for n in G.nodes():
        ccs[n] = nx.clustering(G, n)
        cc_avg = cc_avg+ccs[n]
    cc_avg = cc_avg/len(G.nodes())
    for e in G.edges():
        commonNeighbors = list(
            set(G.neighbors(e[0])).intersection(G.neighbors(e[1])))
        r[e[0], e[1]] = 0
        for neighbor in commonNeighbors:
            C = set(G.neighbors(neighbor))
            C = C.intersection(set(G.neighbors(e[0])))
            C = C.intersection(set(G.neighbors(e[1])))
            # if(C.__contains__(e[0])):
            #    C.remove(e[0])
            # if(C.__contains__(e[1])):
            #    C.remove(e[1])
            r[e[0], e[1]] = r[e[0], e[1]] + \
                len(C)*len(list(G.neighbors(neighbor)))**(-Beta*cc_avg)

    return r


def SRW(G=None, T=5) -> dict:
    r = dict()
    PI = dict()

    for n1 in G.nodes():
        for n2 in G.nodes():
            if (G.has_edge(n1, n2)):
                PI[n1, n2] = 1/len(list(G.neighbors(n1)))
            else:
                PI[n1, n2] = 0

    PI_s = []
    PI_s.append(PI)
    for t in range(0, T):
        PI_tmp = PI.copy()
        for n1 in G.nodes():
            for n2 in G.nodes():
                s = 0
                for nk in G.nodes():
                    s = s+PI[n1, nk]*PI[nk, n2]
                PI_tmp[n1, n2] = s
        PI = PI_tmp
        PI_s.append(PI.copy())

    for e in G.edges():
        r[e[0], e[1]] = 0
        k_x = len(list(G.neighbors(e[0])))/2*len(list(G.edges()))
        k_y = len(list(G.neighbors(e[1])))/2*len(list(G.edges()))
        for pi in range(1, T):
            PI_pi = PI_s[pi]
            r[e[0], e[1]] = r[e[0], e[1]]+k_x * \
                PI_pi[e[0], e[1]]+k_y*PI_pi[e[1], e[0]]

    return r


def HIN(G=None, T=5) -> dict:
    # calculating H-Index
    def H_index(citations):
        # sorting in ascending order
        citations.sort()
        # iterating over the list
        for i, cited in enumerate(citations):
            # finding current result
            result = len(citations) - i
            # if result is less than or equal
            # to cited then return result
            if result <= cited:
                return result
        return 0

    r = dict()
    PI = dict()

    for n1 in G.nodes():
        for n2 in G.nodes():
            if (G.has_edge(n1, n2)):
                PI[n1, n2] = 1/len(list(G.neighbors(n1)))
            else:
                PI[n1, n2] = 0

    H_indexes = dict()
    D_indexes = dict()
    for n in list(G.nodes()):
        n_ds = []
        for neighbor in list(G.neighbors(n)):
            n_ds.append(len(list(G.neighbors(neighbor))))
        H_indexes[n] = H_index(n_ds)
        D_indexes[n] = len(list(G.neighbors(n)))

    H_avg = dict()
    D_avg = dict()
    for n in list(G.nodes()):
        avg_h = 0
        avg_d = 0
        for neighbor in list(G.neighbors(n)):
            avg_d = avg_d+D_indexes[neighbor]
            avg_h = avg_h + H_indexes[neighbor]
        H_avg[n] = avg_h/len(list(G.neighbors(n)))
        D_avg[n] = avg_d / len(list(G.neighbors(n)))

    PI_s = []
    PI_s.append(PI)
    for t in range(0, T):
        PI_tmp = PI.copy()
        for n1 in G.nodes():
            for n2 in G.nodes():
                s = 0
                for nk in G.nodes():
                    s = s+PI[n1, nk]*PI[nk, n2]
                PI_tmp[n1, n2] = s
        PI = PI_tmp
        PI_s.append(PI.copy())

    for e in G.edges():
        r[e[0], e[1]] = 0
        for pi in range(1, T):
            PI_pi = PI_s[pi]
            r[e[0], e[1]] = r[e[0], e[1]]+(math.sqrt(D_avg[e[0]]*H_avg[e[1]]) / 2 * len(list(G.edges())))*PI_pi[e[0], e[1]]+(
                math.sqrt(D_avg[e[1]]*H_avg[e[0]]) / 2 * len(list(G.edges())))*PI_pi[e[1], e[0]]

    return r


def jaffe(G, nodeID1, nodeID2) -> float:
    s1 = 0
    s2 = 0
    C = set(G.neighbors(nodeID1))
    for c in C:
        s1 = s1 + \
            pow(len(set(G.neighbors(nodeID1)).intersection(set(G.neighbors(c)))), 2)
    C = set(G.neighbors(nodeID2))
    for c in C:
        s2 = s2 + \
            pow(len(set(G.neighbors(nodeID2)).intersection(set(G.neighbors(c)))), 2)
    return ((len(list(G.neighbors(nodeID1))) * len(list(G.neighbors(nodeID2)))) / math.sqrt(s2 * s1))


def common_Neighbors(G, nodeID1, nodeID2) -> int:
    return len(set(G.neighbors(nodeID1)).intersection(set(G.neighbors(nodeID2))))


def cosine_similarity(G, nodeID1, nodeID2) -> float:
    return len(set(G.neighbors(nodeID1)).intersection(set(G.neighbors(nodeID2)))) / np.sqrt(
        len(list(G.neighbors(nodeID1))) * len(list(G.neighbors(nodeID2))))


def HPI_similarity(G, nodeID1, nodeID2) -> float:
    return len(set(G.neighbors(nodeID1)).intersection(set(G.neighbors(nodeID2)))) / min(
        len(list(G.neighbors(nodeID1))), len(list(G.neighbors(nodeID2))))


def adamicAdar_similarity(G, nodeID1, nodeID2) -> float:
    C = set(G.neighbors(nodeID1)).intersection(set(G.neighbors(nodeID2)))
    AA = 0
    for c in C:
        AA += 1 / log(len(list(G.neighbors(c))))
    return AA


def RA_similarity(G, nodeID1, nodeID2) -> float:
    C = set(G.neighbors(nodeID1)).intersection(set(G.neighbors(nodeID2)))
    AA = 0
    for c in C:
        AA += 1 / len(list(G.neighbors(c)))
    return AA


def binSearch(A, sInd, eInd, x) -> (int):
    out = 0
    while sInd <= eInd:
        mid = math.floor((sInd + eInd) / 2)
        out = mid
        prev = 0
        if (mid == 0):
            prev = -1
        else:
            prev = A[mid - 1]
        if prev < x and x <= A[mid]:
            out = mid
            break
        elif A[mid - 1] >= x:
            eInd = mid - 1
        else:
            sInd = mid + 1
    return (out)


def crossMutu(mPairs: list, alpha: float) -> list:
    def traverse(sNode: int, sEdge: int, alpha: float, char: list, W: dict) -> dict:
        expanded = [False] * len(char)
        dephthIndex = [0] * len(char)
        cList = []
        if sEdge is not None:
            expanded[sEdge] = True
        for node in list(mst.neighbors(sNode)):
            if (
                char[mst.edges[sNode, node]["index"]] is False
                and expanded[mst.edges[sNode, node]["index"]] is False
            ):
                cList.append(mst.edges[sNode, node]["index"])
                dephthIndex[mst.edges[sNode, node]["index"]] = 1
                expanded[mst.edges[sNode, node]["index"]] = True
        for cItem in cList:
            W[cItem] += alpha * (1 / (2 ** (1 / dephthIndex[cItem]))) + (
                (1 - alpha) * (1 - (1 / (2 ** (1 / dephthIndex[cItem])))))
            endNodes = edgeList[cItem]
            for neighbor in mst.neighbors(endNodes[0]):
                if (
                    expanded[mst.edges[endNodes[0], neighbor]
                             ["index"]] is False
                    and char[mst.edges[endNodes[0], neighbor]["index"]] is False
                ):
                    cList.append(mst.edges[endNodes[0], neighbor]["index"])
                    dephthIndex[mst.edges[endNodes[0], neighbor]
                                ["index"]] = dephthIndex[cItem] + 1
                elif (
                    expanded[mst.edges[endNodes[0], neighbor]
                             ["index"]] is False
                    and char[mst.edges[endNodes[0], neighbor]["index"]] is True
                ):
                    pass
                expanded[mst.edges[endNodes[0], neighbor]["index"]] = True
            for neighbor in mst.neighbors(endNodes[1]):
                if (
                    expanded[mst.edges[endNodes[1], neighbor]
                             ["index"]] is False
                    and char[mst.edges[endNodes[1], neighbor]["index"]] is False
                ):
                    cList.append(mst.edges[endNodes[1], neighbor]["index"])
                    dephthIndex[mst.edges[endNodes[1], neighbor]
                                ["index"]] = dephthIndex[cItem] + 1
                elif (
                    expanded[mst.edges[endNodes[1], neighbor]
                             ["index"]] is False
                    and char[mst.edges[endNodes[1], neighbor]["index"]] is True
                ):
                    pass
                expanded[mst.edges[endNodes[1], neighbor]["index"]] = True
        return W

    def mutuationProb(chr_child: Chromosom, alpha: float, cmId: int, breakList: list) -> dict:
        W = dict.fromkeys(list(range(0, len(edgeList))), 0)
        for breakPoint in breakList:
            if chr_child.cmV[edgeList[breakPoint][0]] == cmId:
                W[breakPoint] += 1
                W = traverse(
                    sNode=edgeList[breakPoint][0], sEdge=breakPoint, alpha=alpha, char=chr_child.m_chr, W=W)
            elif chr_child.cmV[edgeList[breakPoint][1]] == cmId:
                W[breakPoint] += 1
                W = traverse(
                    sNode=edgeList[breakPoint][1], sEdge=breakPoint, alpha=alpha, char=chr_child.m_chr, W=W)
        for node in leafNodes:
            for n_ in list(mst.neighbors(node)):
                if chr_child.cmV[node] == cmId:
                    W[mst.edges[node, n_]["index"]] += 1
                    W = traverse(
                        sNode=node, sEdge=mst.edges[node, n_]["index"], alpha=alpha, char=chr_child.m_chr, W=W)
                elif chr_child.cmV[n_] == cmId:
                    W[mst.edges[node, n_]["index"]] += 1
                    W = traverse(
                        sNode=n_, sEdge=mst.edges[node, n_]["index"], alpha=alpha, char=chr_child.m_chr, W=W)
        S = sum(list(W.values()))
        if S == 0:
            W = dict.fromkeys(list(range(0, len(edgeList))), 1)
            S = len(W)
        for i in list(W.keys()):
            W[i] = W[i] / S
        return W

    popTmp = []
    for pair in mPairs:
        newChr = Chromosom(empty=True)
        breakList = newChr.crossOver(popSpace[pair[0]], popSpace[pair[1]])
        # Mutation phase
        if (mRand.random() > mu_P):
            rn = []
            if (MUTATION_FUNCTION_ID == 1):
                comProb = []
                comCProb = []
                tmp = 0
                min = 0
                for i in range(0, len(newChr.V)):
                    t = newChr.Q / list(newChr.Qs.values())[i]
                    if (t < min):
                        min = t
                    comProb.append(t)
                    tmp = tmp + comProb[i]
                if (min < 0):
                    tmp = 0
                    for i in range(0, len(newChr.V)):
                        comProb[i] = -min + comProb[i]
                        tmp = tmp + comProb[i]
                tmp1 = 0
                for i in range(0, len(newChr.V)):
                    comProb[i] = comProb[i] / tmp
                    tmp1 = tmp1 + comProb[i]
                    comCProb.append(tmp1)

                cmInd = binSearch(comCProb, 0, len(comCProb), mRand.random())
                cmInd = list(newChr.Qs.keys())[cmInd]

                prob = mutuationProb(newChr, alpha, cmInd, breakList)

                cumprob = np.cumsum(list(prob.values()))
                r = binSearch(cumprob, 0, len(cumprob), mRand.random())
                rn.append(list(prob.keys())[r])
            elif (MUTATION_FUNCTION_ID == 2):
                prob = propList
                cumprob = np.cumsum(prob)
                rn.append(binSearch(cumprob, 0, len(cumprob), mRand.random()))
            if (MUTATION_FUNCTION_ID == 0):
                rn.append(mRand.randint(0, len(mst.edges) - 1))
            for r in rn:
                tmp = list(mst.edges).pop(r)
                newChr.mutuate(tmp[0], tmp[1])
        popTmp.append(newChr)
    return popTmp


def chunk_pairs(pairs, chunk_count=4):
    """Yield slices of `pairs` with roughly equal sizes."""
    if chunk_count <= 0:
        yield pairs
        return
    chunk_size = math.ceil(len(pairs) / chunk_count)
    for i in range(chunk_count):
        start = i * chunk_size
        end = min(start + chunk_size, len(pairs))
        if start >= len(pairs):
            break
        yield pairs[start:end]


global mRand
global mst
global G
global G_w
global popSpace
global mu_P
global STEP
global EXPERIMENT_COUNT
global CONVERGENCE_LIMIT
global MUTATION_FUNCTION_ID

if __name__ == '__main__':

    mPath = pathlib.Path("data").resolve()
    onlyfiles = [str(join(mPath, f))
                 for f in listdir(mPath) if isfile(join(mPath, f))]
    c = 0
    mtxFiles = []
    print("************ Finding communities in social networks using the evolutionary method (ver 3)************")
    print("----------------------------------------------------------------------------------------------")
    print("Networks in current directory : ")
    for f in onlyfiles:
        filename, file_extension = os.path.splitext(f)
        if (file_extension == ".mtx"):
            print(str(c) + ")" + f)
            mtxFiles.append(f)
            c += 1
    print("----------------------------------------------------------------------------------------------")
    indMtx = int(input("Enter the ID of the preferred network (int) :"))
    pop_size = int(input("Enter the population size (int) : "))
    T = float(input(
        "Enter the preferred threshold to stop the process after reaching it (float) :"))
    mu_P = float(input("Enter the mutation probability of nodes (float) :"))

    STEP = float(input("Enter stepping parameter for sine function (float) :"))
    EXPERIMENT_COUNT = int(
        input("Enter preferred number of experiments (int) :"))
    CONVERGENCE_LIMIT = int(
        input("Enter preferred limit for convergence [-1: for unlimited] (int) :"))
    MUTATION_FUNCTION_ID = int(input(
        "Enter mutation function ID [0: uniform, 1: sine, 2: weight based] (int) :"))
    print("----------------------------------------------------------------------------------------------")
    mRand = random.Random()
    G: nx.Graph = nx.read_edgelist(mtxFiles[indMtx], delimiter=" ", data=(('weight', float),), nodetype=int,
                                   edgetype=int)

    G.name = os.path.splitext(mtxFiles[indMtx])[0]
    m = 2 * len(G.edges)
    print("Number of edges = " + str(m))
    n = G.number_of_nodes()
    print("Number of nodes = " + str(n))
    print("Max modularity threshold = " + str(T))
    print("Mutation probability = " + str(mu_P))
    print("Initial population size = " + str(pop_size))
    print("STEP = " + str(STEP))
    print("Experiment count = " + str(EXPERIMENT_COUNT))
    print("Convergence Limit = " + str(CONVERGENCE_LIMIT))
    if MUTATION_FUNCTION_ID == 0:
        print("Mutation function = Uniform")
    elif MUTATION_FUNCTION_ID == 1:
        print("Mutation function = Sine")
    elif MUTATION_FUNCTION_ID == 2:
        print("Mutation function = Weight Based")

    print("----------------------------------------------------------------------------------------------")
    G_w = G.copy()
    print("Computing similarity values")

    for edge in G_w.edges:
        j = jaccard(G_w, edge[0], edge[1])
        if (j != 0):
            j = 1 / j
        else:
            j = 9999999999999999999999999999
        G_w.edges[edge[0], edge[1]]["weight"] = j

    print("----------------------------------------------------------------------------------------------")
    print("Finding minimum spanning tree")
    mst: nx.Graph = nx.minimum_spanning_tree(G_w, weight='weight')
    mst.name = "minimum spaning tree of G"
    leafNodes = []
    edgeList = []

    i = 0
    propList = []
    S = 0
    for edge in mst.edges:
        mst.edges[edge[0], edge[1]]["index"] = i
        propList.append((mst.edges[edge[0], edge[1]]["weight"]) ** -1)
        S = S + ((mst.edges[edge[0], edge[1]]["weight"]) ** -1)
        edgeList.append((edge[0], edge[1]))
        if (mst.degree[edge[0]] == 1):
            leafNodes.append(edge[0])
        elif (mst.degree[edge[1]] == 1):
            leafNodes.append(edge[1])
        i += 1
    for i in range(0, len(mst.edges)):
        propList[i] = propList[i] / S

    popSpace = []
    maxsList = []
    qList = []

    for experiment in range(0, EXPERIMENT_COUNT):
        genQs = []
        popSpace.clear()
        print("----------------------------------------------------------------------------------------------")
        print("Initial population generation")
        for i in range(pop_size):
            popSpace.append(Chromosom())
        popSpace.sort(key=lambda x: x.Q, reverse=True)
        print("**********************************************************************************************")
        print("Experiment #" + str(experiment))
        print("**********************************************************************************************")
        fTime = time.time()
        t = 0
        q_old = 0
        q_new = popSpace[0].Q / 2
        convergence = 0
        deg = np.arcsin(0.5)
        while (popSpace[0].Q < T):
            genQs.append(popSpace[0].Q)
            startTime = time.time()
            print('Generation #' + str(t))
            print('popSpace Size = ' + str(len(popSpace)))
            # print("Number of genes to mutuate: " + str(math.ceil(len(mst.edges) * mu_P)))
            t += 1
            if (CONVERGENCE_LIMIT != -1):
                if (convergence > CONVERGENCE_LIMIT):
                    break
            print('Max Q=' + str(popSpace[0].Q))
            x_ = 0
            for i in range(0, len(popSpace)):
                x_ = x_ + popSpace[i].Q
            print('Avg Q=' + str(x_ / len(popSpace)))
            q_old = q_new
            q_new = popSpace[0].Q
            print("degree_old = " + str(deg))
            if (q_new == q_old):
                convergence = convergence + 1
                deg = deg + STEP * np.pi
                # direction = -direction
            else:
                convergence = 0
            print("degree_new = " + str(deg))
            print("Converged in " + str(convergence) + " generations")
            alpha = np.abs(np.sin(deg))
            print("alpha = " + str(alpha))
            # if (popSpace[0].Q == popSpace[pop_size - 1].Q):
            #    #print("No improvement is possible\n\rLast generation didn't have any mutuation")
            #    #break
            #    norm_X = [1 for x in popSpace]
            # else:
            #    norm_X = [(x.Q - popSpace[pop_size - 1].Q) / (popSpace[0].Q - popSpace[pop_size - 1].Q) for x in popSpace]
            norm_X = [x.Q for x in popSpace]
            tmp = sum(norm_X)
            norm_X = [x / tmp for x in norm_X]
            norm_X = np.cumsum(norm_X)
            # popTmp = []
            pairs = []
            for i in range(pop_size - math.floor(pop_size / 2)):
                r1 = mRand.random()
                r2 = mRand.random()
                i1 = binSearch(norm_X, 0, pop_size - 1, r1)
                i2 = binSearch(norm_X, 0, pop_size - 1, r2)
                pairs.append((i1, i2))
            print("process broke to 4 threads")
            popTmp0 = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(crossMutu, chunk, alpha)
                    for chunk in chunk_pairs(pairs, chunk_count=4)
                ]
                for idx, future in enumerate(futures):
                    result = future.result()
                    print(f"t{idx} joined")
                    popTmp0.extend(result)

            popSpace.extend(popTmp0)
            popSpace.sort(key=lambda x: x.Q, reverse=True)
            del popSpace[pop_size:len(popSpace)]

            print(
                "Writing extracted communities inside file :[" + G.name + str('.out]'))
            f = open(G.name + str('.out'), "w")
            f.write(repr(popSpace[0].V))
            f.close()
            print("Elapsed Time: " + str(time.time() - startTime))
            print(
                "----------------------------------------------------------------------------------------------")
        print("Full Elapsed Time: " + str(time.time() - fTime))
        print('Max Q=' + str(popSpace[0].Q))
        f = open(G.name + str(experiment)+str('.out'), "w")
        f.write(repr(genQs))
        f.close()

        f = open(G.name + str('.out'), "w")
        f.write(repr(popSpace[0].V))
        f.close()
        pos = nx.spring_layout(G)
        nx.draw(G, pos=pos)
        for CM in popSpace[0].V:
            color = [
                "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
            nx.draw_networkx_nodes(G=G, pos=pos, nodelist=list(
                popSpace[0].V[CM]), node_color=color, alpha=0.8)
        nx.draw_networkx_labels(G, pos=pos)
        plt.savefig(G.name + str('.out') + ".png")
        plt.close()
        maxsList.append(t)
        qList.append(popSpace[0].Q)
    f = open(G.name + str('.maxs.out'), "w")
    f.write(repr(maxsList))
    f.write(repr(qList))
    f.close()
