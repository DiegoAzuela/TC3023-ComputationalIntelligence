from deap import base, creator, algorithms, tools, gp
import operator
import random
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import math

toolbox = base.Toolbox()

df = pd.DataFrame({'x':list(range(0,10),),
                   'y':list(range(10,0,-1),),
                   'f(x)':[90, 82, 74, 66, 58, 50, 42, 34, 26, 18]})

x = df['x'].values.tolist()
y = df['y'].values.tolist()
z = df['f(x)'].values.tolist()

def eval_func(ind, x, y, outputs):
    func_eval = toolbox.compile(expr=ind)
    predictions = list(map(func_eval,x,y))
    return abs(mean_squared_error(outputs,predictions)),

def div (x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 1

def pot(x,y):
    try:
        return math.pow(x,y)
    except:
        return 1

pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(pot, 2)
pset.addTerminal(math.e, 'e')
pset.addEphemeralConstant('R', lambda: random.randint(1,10))

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=10)
toolbox.register('select', tools.selDoubleTournament, fitness_size=2, parsimony_size=1.4, fitness_first=False)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('mutate', gp.mutNodeReplacement, pset=pset)
toolbox.register('evaluate', eval_func, x=x, y=y, outputs=z)
toolbox.register('compile', gp.compile, pset=pset)


toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('max', np.max)
stats.register('mean', np.mean)
stats.register('std', np.std)

hof = tools.HallOfFame(5)
pop = toolbox.population(n=40)

last_population, log = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.25,ngen=20,stats=stats,halloffame=hof)

for ind in hof:
    print(ind)
    print(toolbox.evaluate(ind, x=x, y=y, outputs=z))
