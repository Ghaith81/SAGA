import numpy as np
import pandas as pd
import deap
from deap import tools
from deap import base, creator
import time
from random import randrange
import copy
import random

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class EvolutionaryWrapperFeatureSelection:

    def HUX(ind1, ind2, fixed=True):
        # index variable
        idx = 0

        # Result list
        res = []

        # With iteration
        for i in ind1:
            if i != ind2[idx]:
                res.append(idx)
            idx = idx + 1
        if (len(res) > 1):
            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            oldInd1 = copy.copy(ind1)

            for i in indx:
                ind1[i] = ind2[i]

            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            for i in indx:
                ind2[i] = oldInd1[i]
        return ind1, ind2

    def evaluate(individual, task, evaluation, dataset, alpha=0.88):
        selected = np.array(individual)
        selectedIndexes = list(np.where(selected == 1)[0])
        c = copy.copy(dataset)
        if (len(selectedIndexes) > 0):
            if (task == 'feature_selection'):
                c.setFeatures(selectedIndexes)
            if (task == 'baseline'):
                c.setInstances(range(0, c.X_train.shape[0]))
                c.setFeatures(range(0, c.X_train.shape[1]))
            c.fitClassifier()
            if (evaluation == 'validation'):
                c.setValidationAccuracy()
                return (alpha * c.getValidationAccuracy()) - ((1 - alpha) * np.sum(selectedIndexes) / len(individual)),
            if (evaluation == 'test'):
                c.setTestAccuracy()
                return (alpha * c.getTestAccuracy()) - ((1 - alpha) * np.sum(selectedIndexes) / len(individual)),

            if (evaluation == 'cv'):
                c.setCV()
                return (alpha * c.getCV()) - ((1 - alpha) * np.sum(selectedIndexes) / len(individual)),
        else:
            return 0,

    def createPopulation(populationSize, indSize, fixedP=0.5):
        pop = []
        for i in range(populationSize):
            if (not fixedP):
                zeroP = random.uniform(0, 1)
            else:
                zeroP = fixedP
            pop.append(deap.creator.Individual(np.random.choice([0, 1], size=(indSize,), p=[zeroP, (1 - zeroP)])))
        return list(pop)

    def createToolbox(indSize, task, evaluation, dataset, alg='CHC', alpha=0.88):
        toolbox = base.Toolbox()
        if (alg == 'CHC'):
            toolbox.register("mate", EvolutionaryWrapperFeatureSelection.HUX)
        elif (alg == 'GA'):
            toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1 / indSize)
        toolbox.register("select", tools.selTournament, tournsize=3)
        if (task == 'feature_selection'):
            toolbox.register("evaluate", EvolutionaryWrapperFeatureSelection.evaluate, task=task, evaluation=evaluation, dataset=dataset, alpha=alpha)
        return toolbox

    def CHC(dataset, population=False, populationSize=40, d=False, divergence=0.35, zeroP=0.5, maxGenerations=np.inf, maxNochange=np.inf, timeout=np.inf,
            task='feature_selection', evaluation='validation', stop=np.inf):
        start = time.time()
        end = time.time()

        task = 'feature_selection'
        indSize = dataset.X_train.shape[1]
        toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, dataset)
        if (not population):
            population = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, zeroP)

        generationCounter = 0
        # calculate fitness tuple for each individual in the population:
        fitnessValues = list(map(toolbox.evaluate, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        best = -1 * np.inf
        noChange = 0
        evaulationCounter = populationSize

        if (d):
            d0 = d
        else:
            d0 = len(population[0])//4
            d = d0

        logDF = pd.DataFrame(
            columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution'))

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            # update counter:
            generationCounter = generationCounter + 1

            # apply the selection operator, to select the next generation's individuals:
            offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, offspring))
            random.shuffle(offspring)

            newOffspring = []

            # apply the crossover operator to pairs of offspring:
            numberOfPaired = 0
            numberOfMutation = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if EvolutionaryWrapperFeatureSelection.hammingDistance(child1, child2) > d:
                    toolbox.mate(child1, child2)
                    numberOfPaired += 1
                    newOffspring.append(child1)
                    newOffspring.append(child2)
                    del child1.fitness.values
                    del child2.fitness.values

            if (d == 0):
                d = d0
                newOffspring = []
                for mutant in offspring:
                    numberOfMutation += 1
                    toolbox.mutate(mutant, indpb=divergence)
                    newOffspring.append(mutant)
                    del mutant.fitness.values

            if (numberOfPaired == 0 and d > 0):
                d -= 1

            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in newOffspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            evaulationCounter = evaulationCounter + len(freshIndividuals)

            if (numberOfMutation == 0):
                population[:] = tools.selBest(population + newOffspring, populationSize)
            else:
                bestInd = tools.selBest(population, 1)
                population[:] = tools.selBest(bestInd + newOffspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            if (best >= maxFitness):
                noChange+= 1
            if (best < maxFitness):
                best = maxFitness
                noChange = 0
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            print("Best Individual = %", np.round(100 * maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            #print()
            #print(np.round(100*maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
            #print()
            end = time.time()
            row = [generationCounter, (end - start), np.round(100 * maxFitness, 2), meanFitness, evaulationCounter,
                   population[best_index]]
            logDF.loc[len(logDF)] = row

        end = time.time()
        return logDF, population

    def GA(dataset, populationSize=40, crossOverP=0.9, mutationP=0.1, zeroP=0.5, maxGenerations=np.inf, maxNochange=np.inf,
            timeout=np.inf,
            task='feature_selection', evaluation='validation', stop=np.inf):

        start = time.time()
        end = time.time()

        task = 'feature_selection'
        indSize = dataset.X_train.shape[1]
        toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, dataset, 'GA')
        population = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, zeroP)

        generationCounter = 0
        # calculate fitness tuple for each individual in the population:
        fitnessValues = list(map(toolbox.evaluate, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        best = -1 * np.inf
        noChange = 0
        evaulationCounter = populationSize

        logDF = pd.DataFrame(
            columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution'))

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            generationCounter = generationCounter + 1

            # apply the selection operator, to select the next generation's individuals:
            offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, offspring))

            # apply the crossover operator to pairs of offspring:

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossOverP:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutationP:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            evaulationCounter = evaulationCounter + len(freshIndividuals)

            population[:] = tools.selBest(population + offspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            if (best >= maxFitness):
                noChange += 1
            if (best < maxFitness):
                best = maxFitness
                noChange = 0
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            print("Best Individual = %", np.round(100 * maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            # print()
            # print(np.round(100*maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
            # print()
            end = time.time()
            row = [generationCounter, (end - start), np.round(100 * maxFitness, 2), meanFitness, evaulationCounter,
                   population[best_index]]
            logDF.loc[len(logDF)] = row

        end = time.time()
        return logDF, population

    def hammingDistance(ind1, ind2):
        ind1 = np.array(ind1)
        ind2 = np.array(ind2)
        return (np.sum(np.abs(ind1 - ind2)))




