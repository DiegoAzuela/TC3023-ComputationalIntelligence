from deap import base, creator, tools
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

fitness = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitness):
    ind.fitness.values = fit

CX = 0.2
mut = 0.1

records = stats.compile(pop)

log = tools.Logbook()
log.record(gen=0, evals=len(pop), **records)

for gen in range(1,10):

    offsprings = toolbox.select(pop, len(pop))

    for child1, child2 in zip(offsprings[::2],offsprings[1::2]):
        if random.random()<CX:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for child in offsprings:
        if random.random()<mut:
            toolbox.mutate(child)
            del child.fitness.values

    invalid_fitness = [ind for ind in offsprings if not ind.fitness.valid]
    fitness = map(toolbox.evaluate, invalid_fitness)
    for ind, fit in zip(invalid_fitness, fitness):
        ind.fitness.values = fit

    pop[:] = offsprings

    records = stats.compile(pop)
    log.record(gen=gen, evals=len(invalid_fitness), **records)
    print(log.stream)