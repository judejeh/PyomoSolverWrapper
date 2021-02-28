# ---------------------------------------------------------------------------------
# Pyomo Solver Wrapper
# Language - Python
# https://github.com/judejeh/PyomoSolverWrapper
# Licensed under MIT license
# Copyright 2021 The Pyomo Solver Wrapper authors <https://github.com/judejeh>
# ---------------------------------------------------------------------------------

# *********************************************************************************
#   Data pre-processing for layout models
#   Contains methods for pre-processing data for
#     for layout models in the 'models' package
#
#   @author Jude E.
#   02.2020
# *********************************************************************************

import numpy as np

def layout98_ex0():
    data_dict = dict()

    data_dict['items'] = list(np.arange(1,8))

    data_dict['connected_items'] = [(1, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7)]

    data_dict['nonoverlap_pairs'] = []
    for i in range(0,len(data_dict['items'])):
        j = i + 1
        while j <= len(data_dict['items'])-1:
            data_dict['nonoverlap_pairs'].append((data_dict['items'][i], data_dict['items'][j]))
            j += 1

    data_dict['alpha'] = {1: 5.22, 2: 11.42, 3: 7.68, 4: 8.48, 5: 7.68, 6: 2.60, 7: 2.40}

    data_dict['beta'] = {1: 5.22, 2: 11.42, 3: 7.68, 4: 8.48, 5: 7.68, 6: 2.60, 7: 2.40}

    data_dict['connection_cost'] = {(1,2): 346.0, (1,5): 416.3, (2,3): 118.0, (3,4): 111.0, (4,5): 85.3, (5,6): 86.3,
                                    (5,7): 82.8, (6,7): 6.5}

    data_dict['BM'] = 100

    # Return
    return data_dict

