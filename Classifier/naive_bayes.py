import numpy as np
import pandas as pd
import sklearn.datasets as skdata


def naive_bayes(X, y, x):
    m = len(X.index)
    A_continue = [a for a in X.columns if len(X[a].value_counts()) > 10]

    A_counts = {}
    for a in X.columns:
        if a in A_continue:
            continue
        A_counts[a] = X[a].value_counts(sort=False)

    A_p = {}
    for counts in A_counts.keys():
        A_p[counts] = A_counts[counts] / m
    print(A_p)
    X = pd.DataFrame(X)
    X["label"] = y
    C_counts = X["label"].value_counts(sort=False)
    C_p = (C_counts+1) / (m+len(X["label"].value_counts(sort=False).index))
    print(C_p)
    C_x_p = []
    for c in C_counts.index:
        t_counts = []
        m_c = C_counts[c]
        for a in A_counts.keys():
            tt_counts = []
            for i in X[a].value_counts().index:
                tt_counts.append((len(X[(X["label"] == c) & (X[a] == i)].index) + 1) / (m_c + len(X[a].value_counts().index)))
            t_counts.append(tt_counts)
        C_x_p.append(t_counts)
    print(C_x_p)

    ys = []
    for j in x.index:
        xx = x.iloc[j]
        pc = {}
        t = 0
        for c in C_counts.index:
            p = 1.0
            tt = 0
            for a in A_counts.keys():
                ttt = 0
                for i in X[a].value_counts().index:
                    if xx[a] == i:
                        p *= C_x_p[t][tt][ttt] / A_p[a][i]
                    ttt += 1
                tt += 1
            t += 1
            pc[c] = p * C_p[c]
        ys.append(pc)
    return ys


if __name__ == "__main__":
    data_x, data_y = [[1,2,1],[1,1,2],[1,1,1],[2,1,1],[1,1,2],[2,1,2],[2,2,1],[1,1,1],[2,1,1]], [1,1,1,2,2,2,3,3,3]
    data_x = pd.DataFrame(data_x)
    x = pd.DataFrame([[2,2,2], [1,1,1]])
    ys = naive_bayes(data_x, data_y, x)
    print(ys)