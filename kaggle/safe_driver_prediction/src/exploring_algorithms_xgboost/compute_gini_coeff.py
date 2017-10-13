
def compute_gini_coeff(x,y):
    '''
    gini coefficient was introduced in economics to quantify income inequality in a society.

    :param x: property whose cumsum will be on x-axis, if it is 0, 1, 2, ..., x = [1, 1, 1, ...]
    :param y: property whose cumsum will be on y-axis, if it is 0, 1, 2, ..., x = [1, 1, 1, ...]
    :return: Gini coefficient between 0 and 1
    '''
    y1, y2 = [], []
    sum_x, sum_y = 0, 0
    for i in range(len(x)):
        sum_x += x[i]
        y1.append(sum_x)
    for i in range(len(y)):
        sum_y += y[i]
        y2.append(sum_y)
    for i in range(len(x)):
        y1[i] = y1[i] - x[0]
    for i in range(len(y)):
        y2[i] = y2[i] - y[0]

    # Area under Lorentz curve
    B12, B21 = 0, 0
    for i in range(len(x)-1):
        B12 += ( y1[i+1] + y1[i] ) * ( y2[i+1] - y2[i] ) / 2
        B21 += ( y2[i + 1] + y2[i] ) * (y1[i + 1] - y1[i]) / 2
    R = ( y1[-1] - y1[0] ) * ( y2[-1]-y2[0] ) / 2
    A12, A21 = R - B12, R - B21
    if A12 >= 0:
        G = A12/R
    else:
        G = A21/R
    return G



if __name__ == '__main__':
    x = [1, 1, 1, 1, 1, 1, 1, 1]
    y = [1, 1, 1, 1, 1, 1, 1, 1]
    #print( gini(x, y) == 0)
    print('G =', gini(x, y))


    x = [1, 1, 1, 1, 1, 1, 1, 1]
    y = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
    #print( gini(x, y) == 0)
    print('G  = ', gini(x, y), 'should be more than 0.5')


    x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    #print( gini(x, y) == 0)
    print('G  = ', gini(x, y), 'should be close to 1, note <1 since edge effect in area')


    x = [1, 1, 1, 1, 1, 1, 1, 1]
    y = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    #print( gini(x, y) == 0)
    print('G  = ', gini(x, y), 'should be less than 0.5')

