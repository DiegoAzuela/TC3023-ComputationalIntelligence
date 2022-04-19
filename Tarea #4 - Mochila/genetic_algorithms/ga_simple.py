from deap import base, creator, tools, algorithms
import numpy as np
import random

def func_eval(ind, ):
    return int("".join(map(str,ind)), base=2),


toolbox = base.Toolbox()

toolbox.register('select', tools.selRoulette)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.1)
toolbox.register('evaluate', func_eval)

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox.register('bit', random.randint, a=0, b=1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.bit, n=5)


toolbox.register('population', tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('max', np.max)
stats.register('mean', np.mean)
stats.register('std', np.std)
stats.register('median', np.median)

pop = toolbox.population(n=10)

hof = tools.HallOfFame(3)

population, log = algorithms.eaSimple(pop,toolbox,1.0,0.1,10,stats=stats, halloffame=hof)

print(repr(log))
print(hof)