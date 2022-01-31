# ---------------------------------------------------------------------------------
# Pyomo Solver Wrapper
# Language - Python
# https://github.com/judejeh/PyomoSolverWrapper
# Licensed under MIT license
# Copyright 2021 The Pyomo Solver Wrapper authors <https://github.com/judejeh>
# ---------------------------------------------------------------------------------

# ********************************************************************************************
#                               PYOMO optimization model                                    #
# ********************************************************************************************
# Type:     Single floor layout model                                                       #
# Source:   Papageorgiou, Lazaros G., and Guillermo E. Rotstein. "Continuous-domain
#           mathematical models for optimal process plant layout." Industrial &
#           engineering chemistry research 37.9 (1998): 3631-3639.
# Author:   Jude E.
# ********************************************************************************************

from pyomo.environ import *

def single_floor_MILP_model(model_data):
    # Create concrete model
    model = ConcreteModel('layout98 model')

    # Define model setsy
    model.i = Set(initialize=model_data['items'], doc='items', ordered=1)
    model.j = Set(within=model.i, initialize=model_data['items'], doc='other items', ordered=1)
    model.ij = Set(within=model.i*model.j, initialize=model_data['connected_items'], doc='connected items')


    # Define parameters
    model.alpha = Param(model.i, initialize=model_data['alpha'], \
                        doc='geometry of item i 1')
    model.beta = Param(model.i, initialize=model_data['beta'], doc='geometry of item i 2')
    model.Cc = Param(model.ij, initialize=model_data['connection_cost'], doc='connection cost')
    model.BM = Param(initialize=model_data['BM'], doc="big 'M' value")

    # Define binary variables
    model.O = Var(model.i, within=Binary, doc='equipment orientation binary')
    model.E1 = Var(model.i, model.j, within=Binary, doc='non overlapping binary variable 1')
    model.E2 = Var(model.i, model.j, within=Binary, doc='non overlapping binary variable 2')

    # Define continuous variables
    model.l = Var(model.i, within=PositiveReals, doc='length of item i')
    model.d = Var(model.i, within=PositiveReals, doc='breadth of item i')
    model.x = Var(model.i, within=PositiveReals, doc='x coordinate for item i')
    model.y = Var(model.i, within=PositiveReals, doc='y coordinate for item i')
    model.R = Var(model.i, model.j, within=PositiveReals, doc='relative distance between item i and j if i is RHS of j')
    model.G = Var(model.i, model.j, within=PositiveReals, doc='relative distance between item i and j if i is LHS of j')
    model.A = Var(model.i, model.j, within=PositiveReals, doc='relative distance between item i and j if i is above j')
    model.B = Var(model.i, model.j, within=PositiveReals, doc='relative distance between item i and j if i is below j')
    model.TD = Var(model.i, model.j, within=PositiveReals, doc='relative distance between item i and j')


    # Define constraints
    ## Orientation constraint 01
    def orientation_rule_1(model, i):
        return model.l[i] == model.alpha[i]*model.O[i] + model.beta[i]*(1 - model.O[i])

    model.OrientationConst01 = Constraint(model.i, rule=orientation_rule_1, doc='Orientation Constraint 1')

    ## Orientation constraint 02
    def orientation_rule_2(model, i):
        return model.d[i] == model.alpha[i] + model.beta[i] - model.l[i]

    model.OrientationConst02 = Constraint(model.i, rule=orientation_rule_2, doc='Orientation Constraint 2')


    ## Distance constraint 01
    def distance_rule_1(model, i, j):
        if (i, j) in model_data['connected_items']:
            return model.R[i, j] - model.G[i, j] == model.x[i] - model.x[j]
        else:
            return Constraint.Skip

    model.DistanceConstr_01 = Constraint(model.i, model.j, rule=distance_rule_1, doc='Distance Constraint 1')

    ## Distance constraint 02
    def distance_rule_2(model, i, j):
        if (i, j) in model_data['connected_items']:
            return model.A[i, j] - model.B[i, j] == model.y[i] - model.y[j]
        else:
            return Constraint.Skip

    model.DistanceConstr_02 = Constraint(model.i, model.j, rule=distance_rule_2, doc='Distance Constraint 2')

    ## Distance constraint 03
    def distance_rule_3(model, i, j):
        if (i, j) in model.ij:
            return model.TD[i,j] == model.R[i, j] + model.G[i, j] + model.A[i, j] + model.B[i, j]
        else:
            return Constraint.Skip

    model.DistanceConstr_03 = Constraint(model.i, model.j, rule=distance_rule_3, doc='Distance Constraint 3')


    ## Non overlapping constraint 01
    def NOLP_rule_1(model, i, j):
        if (i, j) in model_data['connected_items']:
            return model.x[i] - model.x[j] + model.BM * (model.E1[i, j] + model.E2[i, j]) >= model.l[i] / 2 + model.l[j] / 2
        else:
            return Constraint.Skip

    model.NOLP_1 = Constraint(model.i, model.j, rule=NOLP_rule_1,
                              doc='Non overlapping constraint 1')

    ## Non overlapping constraint 02
    def NOLP_rule_2(model, i, j):
        if (i, j) in model_data['connected_items']:
            return model.x[j] - model.x[i] + model.BM * (1 - model.E1[i, j] + model.E2[i, j]) >= model.l[i] / 2 + model.l[
                j] / 2
        else:
            return Constraint.Skip

    model.NOLP_2 = Constraint(model.i, model.j, rule=NOLP_rule_2,
                              doc='Non overlapping constraint 2')

    ## Non overlapping constraint 03
    def NOLP_rule_3(model, i, j):
        if (i, j) in model_data['connected_items']:
            return model.y[i] - model.y[j] + model.BM * (1 + model.E1[i, j] - model.E2[i, j]) >= model.d[i] / 2 + model.d[
                j] / 2
        else:
            return Constraint.Skip

    model.NOLP_3 = Constraint(model.i, model.j, rule=NOLP_rule_3,
                              doc='Non overlapping constraint 3')

    ## Non overlapping constraint 04
    def NOLP_rule_4(model, i, j):
        if (i, j) in model_data['connected_items']:
            return model.y[j] - model.y[i] + model.BM * (2 - model.E1[i, j] - model.E2[i, j]) >= model.d[i] / 2 + model.d[
                j] / 2
        else:
            return Constraint.Skip

    model.NOLP_4 = Constraint(model.i, model.j, rule=NOLP_rule_4,
                              doc='Non overlapping constraint 4')


    ## Layout design constraint 01
    def LDC_rule_1(model, i):
        return model.x[i] >= model.l[i]/2

    model.LDC_1 = Constraint(model.i, rule=LDC_rule_1, doc='Layout design constraint 1')

    ## Layout design constraint 02
    def LDC_rule_2(model, i):
        return model.y[i] >= model.d[i] / 2

    model.LDC_2 = Constraint(model.i, rule=LDC_rule_2, doc='Layout design constraint 2')

    # Define objective function
    model.obj = Objective(expr=sum(model.TD[i, j]*model.Cc[i,j] for (i, j) in model.ij), sense=minimize,
                          doc='Minimise the total connection distance')


    return model

