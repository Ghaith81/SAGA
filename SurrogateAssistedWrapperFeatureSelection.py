import numpy as np
from EvolutionaryWrapperFeatureSelection import EvolutionaryWrapperFeatureSelection
import time
import pandas as pd
import numpy as np
import copy
import random
from deap import tools

class SurrogateAssistedWrapperFeatureSelection:
    def SAGA(dataset, populationSize=40, a=16, reductionRate=0.5, step=10, d=10, zeroP=0.5,
             verbose=False, qualOnly=False, timeout=np.inf, noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations',
                                       'best_solution', 'best_fitness_original'))
        partialDataset = copy.copy(dataset)
        sampleSize = partialDataset.X_train.shape[0] // a
        randomSampling = random.sample(range(partialDataset.X_train.shape[0]), sampleSize)
        partialDataset.setInstances(randomSampling)
        if (verbose):
            print('Current Approx Sample Size:', len(partialDataset.instances))
            print('Current Population Size:', populationSize)
        task = 'feature_selection'
        indSize = partialDataset.X_train.shape[1]
        toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, partialDataset)
        population = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, zeroP)

        bestTrueFitnessValue = -1 * np.inf
        sagaFeatureSubset = [1] * (len(partialDataset.features))
        qual = False

        numberOfEvaluations = 0
        generationCounter = 0
        maxAllowedSize = int(partialDataset.X_train.shape[0])

        while True:
            #toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, partialDataset)
            log, population = EvolutionaryWrapperFeatureSelection.CHC(partialDataset,
                                                                      population,
                                                                      populationSize=populationSize,
                                                                      maxGenerations=step,
                                                                      task=task)

            generationCounter = generationCounter + step
            featureIndividual = log.iloc[-1]['best_solution']

            # Check if SAGA identified new feature subset

            if (sagaFeatureSubset != featureIndividual):
                trueBestInGeneration = np.round(
                    100 * EvolutionaryWrapperFeatureSelection.evaluate(featureIndividual, 'feature_selection', evaluation, dataset)[0], 2)
                approxBestInGeneration = np.round(
                    100 * EvolutionaryWrapperFeatureSelection.evaluate(featureIndividual, 'feature_selection', evaluation, partialDataset)[0], 2)
                numberOfEvaluations += 1
                end = time.time()
                row = [generationCounter, (end - start), approxBestInGeneration, 'NA',
                       numberOfEvaluations, featureIndividual, trueBestInGeneration]
                if (verbose):
                    print(row)

                # Check if the original value improved
                if (trueBestInGeneration > bestTrueFitnessValue):
                    bestTrueFitnessValue = trueBestInGeneration
                    sagaFeatureSubset = featureIndividual
                    sagaIndividual = tools.selBest(population, 1)
                    if (verbose):
                        print('The best individual is saved', bestTrueFitnessValue)
                        print('Number of features in selected individual: ', np.sum(sagaFeatureSubset))
                    row = [generationCounter, (end - start), approxBestInGeneration, 'NA',
                           numberOfEvaluations, sagaFeatureSubset, bestTrueFitnessValue]
                    logDF.loc[len(logDF)] = row

                # A possible false optimum is detected.

                elif (len(partialDataset.instances) < maxAllowedSize):
                    if (verbose):
                        print('A possible false optimum is detected!')
                    best_approx_fitness_value = 0
                    a = a / 2
                    sampleSize = int(partialDataset.X_train.shape[0] / a)
                    populationSize = int(populationSize * reductionRate)
                    if (sampleSize < maxAllowedSize):
                        randomSampling = random.sample(range(partialDataset.X_train.shape[0]), sampleSize)
                        partialDataset.setInstances(randomSampling)
                        newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, zeroP)

                        population[:] = tools.selBest(sagaIndividual + newInd, populationSize)
                        if (verbose):
                            print('Current Approx Sample Size:', len(partialDataset.instances))
                            print('Current Population Size:', populationSize)

                    else:
                        if (qualOnly):
                            end = time.time()
                            qualTime = (end - start)
                            return logDF
                        partialDataset = copy.copy(dataset)
                        onesP = (np.sum(sagaIndividual) / indSize)
                        populationSize = 40
                        newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, 1 - onesP)
                        population[:] = tools.selBest(sagaIndividual + newInd, populationSize)

                        sagaFeatureSubset = copy.copy(dataset)
                        toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, sagaFeatureSubset)
                        if (verbose):
                            print('Approximation stage is over!')
                            print('Current Approx Sample Size:', len(partialDataset.instances))
                            print('Current Population Size:', populationSize)

                        end = time.time()
                        qualTime = (end - start)
                        log, population = EvolutionaryWrapperFeatureSelection.CHC(dataset,
                                                                                  population,
                                                                                  populationSize=populationSize,
                                                                                  maxNochange=noChange,
                                                                                  timeout=timeout-qualTime,
                                                                                  task=task)

                        break

                elif (len(partialDataset.instances) >= maxAllowedSize):
                    break

            # The current approximation converged!
            else:

                # Check if the current appoximation is the maximum allowed.

                if (len(partialDataset.instances) >= maxAllowedSize):
                    break

                if (verbose):
                    print('The approximation converged!')
                best_approx_fitness_value = 0
                a = a / 2
                sampleSize = int(partialDataset.X_train.shape[0] / a)
                populationSize = int(populationSize * reductionRate)
                if (sampleSize < maxAllowedSize):
                    randomSampling = random.sample(range(partialDataset.X_train.shape[0]), sampleSize)
                    partialDataset.setInstances(randomSampling)
                    newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, zeroP)

                    population[:] = tools.selBest(sagaIndividual + newInd, populationSize)
                    if (verbose):
                        print('Current Approx Sample Size:', len(partialDataset.instances))
                        print('Current Population Size:', populationSize)

                else:
                    if (qualOnly):
                        end = time.time()
                        qualTime = (end - start)
                        return logDF
                    partialDataset = copy.copy(dataset)
                    onesP = (np.sum(sagaIndividual) / indSize)
                    populationSize = 40
                    newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, 1 - onesP)
                    population[:] = tools.selBest(sagaIndividual + newInd, populationSize)
                    partialDataset = copy.copy(dataset)
                    toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, partialDataset)
                    if (verbose):
                        print('Approximation stage is over!')
                        print('Current Approx Sample Size:', len(partialDataset.instances))
                        print('Current Population Size:', populationSize)
                    end = time.time()
                    qualTime = (end - start)

                    log, population = EvolutionaryWrapperFeatureSelection.CHC(dataset,
                                                                              population,
                                                                              populationSize=populationSize,
                                                                              maxNochange=noChange,
                                                                              timeout=timeout - qualTime,
                                                                              task=task)

                    break

        for index, row in log.iterrows():
            log.at[index, 'time'] = log.loc[index, 'time'] + qualTime
            log.at[index, 'number_of_evaluations'] = log.loc[index, 'number_of_evaluations'] + numberOfEvaluations
            log.at[index, 'generation'] = log.loc[index, 'generation'] + logDF.iloc[-1]['generation']
        log['best_fitness_original'] = log['best_fitness']
        logDF = pd.concat((logDF, log))

        return logDF, population


