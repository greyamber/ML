# coding: utf-8
import numpy as np
import sklearn.datasets as skdata
import pandas as pd


class Node(object):
    def __init__(self, xs):
        self.left = self.right = None
        self.lambda_function = None
        self.xs = xs
        self.classify = None


def find_max_gini(node, A_continue):
    Gini_index = {}
    max_feature = None
    max_gini = -1
    break_flag = []

    now_conuts = node.xs['label'].value_counts(sort=False)
    node_gini = 1 - np.sum(np.array(now_conuts / now_conuts.sum()) ** 2)
    for a in node.xs.columns:
        if a == 'label':
            continue
        Xl = Xr = None
        if a in A_continue:
            value = node.xs[a].median()
            Xl = node.xs[node.xs[a] >= value]
            Xr = node.xs[node.xs[a] < value]
            lambda_function = (a, value, "c")
        else:
            values = node.xs[a].value_counts().index
            Xl = node.xs[node.xs[a] == values[0]]
            Xr = node.xs[node.xs[a] != values[0]]
            lambda_function = (a, values[0], "d")
        break_flag.append(Xl.empty or Xr.empty)

        countsl = Xl['label'].value_counts(sort=False)
        countsr = Xr['label'].value_counts(sort=False)

        Ginil = 1 - np.sum(np.array(countsl / countsl.sum()) ** 2)
        Ginir = 1 - np.sum(np.array(countsr / countsr.sum()) ** 2)
        Gini_index[a] = [node_gini - (Ginil * len(Xl.index) + Ginir * len(Xr.index)) / float(len(node.xs.index)),
                         Xl, Xr, lambda_function]
        if Gini_index[a][0] > max_gini:
            max_feature = a
            max_gini = Gini_index[a][0]

    return node_gini, Gini_index, max_feature, max_gini, break_flag, Gini_index[max_feature][3]


def cart(X, y, mission_type, stop_gini):
    stack = []
    A_continue = [a for a in X.columns if len(X[a].value_counts()) > 10]
    A_multi_classes = [a for a in X.columns if (len(X[a].value_counts()) > 2 and a not in A_continue)]
    # one-hot
    for a in A_multi_classes:
        c = X[a].value_counts()
        for i in c.index:
            X["__" + str(a) + "_one_hot_plus_" + str(i)] = [1 if row == i else 0 for row in X[a]]
        del X[a]
    X = pd.DataFrame(X)
    X["label"] = y
    root = node = Node(X)
    if_end = False
    while True:
        # compute a and get sub xs_l, xs_r
        # if---:
        #   node.classify=label
        #   while True:
        #       node=stack.pop().right
        #       if---: node.classify = label
        #       else: break
        # else: stack.append(node), node.left = Node(), node.right = Node(), node = node.left
        node_gini, Gini_index, max_feature, max_gini, break_flag, lambda_function = find_max_gini(node, A_continue)

        if node_gini <= stop_gini or all(break_flag):
            node.classify = node.xs["label"].value_counts().argmax()
            while True:
                try:
                    node = stack.pop().right
                except Exception as e:
                    if_end = True
                    break
                node_gini, Gini_index, max_feature, max_gini, break_flag, lambda_function = find_max_gini(node, A_continue)

                if node_gini <= stop_gini or all(break_flag):
                    node.classify = node.xs["label"].value_counts().argmax()
                else:
                    node.lambda_function = lambda_function
                    break
        else:
            node.lambda_function = lambda_function
            stack.append(node)
            node.left = Node(Gini_index[max_feature][1])
            node.right = Node(Gini_index[max_feature][2])
            node = node.left

        if if_end:
            break

    return root, A_continue


def predict(root, X):
    y = []
    for i in range(len(X.index)):
        x = X.iloc[i]
        node = root
        while True:
            if node.lambda_function is None:
                y.append(node.classify)
                break
            lambda_tuple = node.lambda_function
            if lambda_tuple[2] == "c":
                if x[lambda_tuple[0]] >= lambda_tuple[1]:
                    node = node.left
                else:
                    node = node.right
            else:
                if x[lambda_tuple[0]] == lambda_tuple[1]:
                    node = node.left
                else:
                    node = node.right
    return np.array(y)


def check_dt(root):
    if root.left is not None:
        check_dt(root.left)
    if root.right is not None:
        check_dt(root.right)
    if root.left is None and root.right is None:
        print(root.xs)


if __name__ == "__main__":
    data_x, data_y = skdata.load_breast_cancer(return_X_y=True)
    data_x = pd.DataFrame(data_x)
    root, A_continue = cart(data_x, data_y, None, 0.0)
    print(predict(root, data_x.iloc[0:150]))
    print(data_y[0:150])
    print(np.sum(np.abs(predict(root, data_x.iloc[0:150]) - data_y[0:150])))
    e = np.sum(np.abs(predict(root, data_x.iloc[0:150]) - data_y[0:150])) / len(data_y[0:150])
    print(1-e)









