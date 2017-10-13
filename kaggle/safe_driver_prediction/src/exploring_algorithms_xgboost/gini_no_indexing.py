def gini(y, pred):
    import numpy as np
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)

    g = g[np.lexsort((g[:,2], -1*g[:,1]))] # sort descending on pred column first, then ascending on indices
    gs = g[:,0].cumsum().sum() / g[:,0].sum()

    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)


def gini2(solution, submission):
    df = zip(solution, submission, range(len(solution)))

    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    print(df)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    print(rand)
    totalPos = float(sum([x[0] for x in df]))
    print(totalPos)
    cumPosFound = [df[0][0]]
    print(cumPosFound)
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini2(solution, submission):
    if gini(solution, solution) == 0:
        normalized_gini = -10
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini



if __name__ == '__main__':
    y = [1, 1, 1, 1 ]
    pred = [1, 1, 1, 1]
    #print( gini(x, y) == 0)
    print('G =', gini(y, pred))
   # print('xgb_g', gini_xgb(y, pred))
    print('G2 =', gini2(y, pred))
    print('normg', normalized_gini2(y, pred) )
    # x = [1, 1, 1, 1, 1, 1, 1, 1]
    # y = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
    # #print( gini(x, y) == 0)
    # print('G  = ', gini(x, y), 'should be more than 0.5')
    #
    # x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # # print( gini(x, y) == 0)
    # print('G  = ', gini(x, y), 'should be close to 1, note <1 since edge effect in area')
    #
    # x = [1, 1, 1, 1, 1, 1, 1, 1]
    # y = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    # # print( gini(x, y) == 0)
    # print('G  = ', gini(x, y), 'should be less than 0.5')


