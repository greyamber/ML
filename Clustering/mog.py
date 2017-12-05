# coding: utf-8
# Mixture of Gaussian
import numpy as np
import Lance.Lance_ML.Clustering.k_means as km


def g(u, c, x, n, la=1e-6):
    x = x - u
    x = np.matrix(x)
    c = np.matrix(c)
    c = c + la * np.eye(n)
    exp = np.exp(-0.5*(np.dot(np.dot(x, c.I), x.T)))
    p = exp / np.sqrt((2*np.pi)**n * np.abs(np.linalg.det(c)))
    return p[0,0]


def mog(X, k, c, step):
    m, n = X.shape
    paras = [[1.0/float(len(k)), k[i], c[i]] for i in range(len(k))]

    if step == 0:
        gama = np.array([[g(para[1], para[2], X[j], n) for j in range(m)] for para in paras])
        sum_gama = np.sum([gama[i] * paras[i][0] for i in range(len(paras))], 0)
        gama_f = [gama[i] * paras[i][0] / (sum_gama + 1e-9) for i in range(len(gama))]

    for _ in range(step):
        gama = np.array([[g(para[1], para[2], X[j], n) for j in range(m)] for para in paras])
        sum_gama = np.sum([gama[i] * paras[i][0] for i in range(len(paras))], 0)
        gama_f = [gama[i] * paras[i][0] / (sum_gama + 1e-9) for i in range(len(gama))]
        for i in range(len(gama_f)):
            paras[i][0] = np.sum(gama_f[i]) / m  # / np.sum(gama_f)
            paras[i][1] = np.sum(np.reshape(gama_f[i], [-1,1]) * X, 0) / np.sum(gama_f[i])

            paras[i][2] = np.sum([gama_f[i][j] * np.dot(np.matrix(X[j]-paras[i][1]).T, np.matrix(X[j]-paras[i][1]))
                                  for j in range(m)], 0) / np.sum(gama_f[i])

    argmax = np.argmax(np.transpose(gama_f, [1,0]), 1)

    return argmax, paras


if __name__ == "__main__":
    X = np.random.random([10,5])
    X = np.concatenate((X, X+20, X*20), axis=0)
    _, ks = km.k_means(X)

    argmax, paras = mog(X, ks, [np.eye(X.shape[1])]*len(ks), 100)
    print("arg:", argmax)
    print("paras:", paras)