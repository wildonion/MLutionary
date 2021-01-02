

# coding: utf-8

'''

	Codded By : 

 █     █░ ██▓ ██▓    ▓█████▄  ▒█████   ███▄    █  ██▓ ▒█████   ███▄    █ 
▓█░ █ ░█░▓██▒▓██▒    ▒██▀ ██▌▒██▒  ██▒ ██ ▀█   █ ▓██▒▒██▒  ██▒ ██ ▀█   █ 
▒█░ █ ░█ ▒██▒▒██░    ░██   █▌▒██░  ██▒▓██  ▀█ ██▒▒██▒▒██░  ██▒▓██  ▀█ ██▒
░█░ █ ░█ ░██░▒██░    ░▓█▄   ▌▒██   ██░▓██▒  ▐▌██▒░██░▒██   ██░▓██▒  ▐▌██▒
░░██▒██▓ ░██░░██████▒░▒████▓ ░ ████▓▒░▒██░   ▓██░░██░░ ████▓▒░▒██░   ▓██
 -------------------------------------------------------------------------------------------------
| Feature Selection and Dimensionality Reduction using Genetic Algorithm For Breast Cancer Dataset
|-------------------------------------------------------------------------------------------------
|
| USAGE : 
|			python bfs_dr.py --chromosomes 200 --features 30 \
|				 			 --generation 3 --parents 10 --selection-method rank \
|				 			 --crossover-method multi_point --mutation-method flipping --mutation-rate 0.20
|
|
|
|
| AVAILABLE GA OPERATORS METHODS: :
| 									selection methods -> roulette_wheel, tournament, rank
|									crossover methods -> multi_point, single_point, two_point
|									mutation methods  -> filipping, reversing, interchanging
|
|
|

'''

import os
import numpy as np
import argparse
from _evolver import population, genetic_process
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ------------ argument options
# ------------------------------
parser = argparse.ArgumentParser(description='Feature Selection of Breast Cancer Dataset using Genetic Algorithm')
parser.add_argument('--chromosomes', action='store', type=int, help='The number of chromosomes in a population', required=True)
parser.add_argument('--features', action='store', type=int, help='The number of total features in dataset as genes must be optimized during each generation', required=True)
parser.add_argument('--generation', action='store', type=int, help='The number of generation', required=True)
parser.add_argument('--parents', action='store', type=int, help='The number of parents to mate for offspring', required=True)
parser.add_argument('--selection-method', action='store', type=str, help='Selection method for crossover operation (roulette_wheel, tournament or rank)', required=True)
parser.add_argument('--crossover-method', action='store', type=str, help='Crossover method to generate offspring (single_point, two_point or multi_point)', required=True)
parser.add_argument('--mutation-method', action='store', type=str, help='Mutation method to mutate offspring (flipping, reversing or interchanging)', required=True)
parser.add_argument('--mutation-rate', action='store', type=float, help='Mutation rate', required=True)
args = parser.parse_args()

BEST_CHROMOSOME_PATH = f"best_chromo_in_{args.generation}_generations.npy"
BEST_SCORE_PATH = f"best_scores_in_{args.generation}_generations.npy"


# ------------ load dataset and split it into training and testing set
# ---------------------------------------------------------------------
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
label = cancer["target"]
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
df = pd.DataFrame(scaled_df, columns=df.columns)
x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=0.30, random_state=101) # [398 rows x 30 columns] for x_train
logmodel = LogisticRegression()



# ------------ prediction on total features of dataset
# -----------------------------------------------------
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)
print(f"▶ Accuracy before GA = {accuracy_score(y_test, predictions)}")



# ------------ prediction on 30 features of dataset selected using GA
# --------------------------------------------------------------------
data = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
pop = population(args.chromosomes, args.features)
# print(f"second index of third population genes with length {len(pop[3])} is ::: {pop[3].genes_objects[2].allele}") # >>>>>>>>>>>> testing design pattern


if os.path.exists(BEST_CHROMOSOME_PATH) and os.path.exists(BEST_SCORE_PATH): # load saved chromosomes and scores
	best_chromosomes = np.load(BEST_CHROMOSOME_PATH)
	best_scores = np.load(BEST_SCORE_PATH)
else: # create a genetic process
	app = genetic_process(generation=args.generation, population=pop, parents=args.parents, model=logmodel, selection_method=args.selection_method, 
					  	  crossover_method=args.crossover_method, mutation_method=args.mutation_method, mutation_rate=args.mutation_rate, **data)
	app.run() # run the process
	app.plot() # plot result
	app.save() # save best chromosomes and scores
	best_chromosomes = app.best_chromosomes # all best chromosomes in every generation
	best_scores = app.best_scores # all best scores in every generation




for i in range(len(best_scores)):
	print(f"🔬 Generation {i} Score --- {best_scores[i]:.0%}")
print()
print('▶ Average accepted score = ', np.mean(best_scores))
print('▶ Median score for accepted scores = ', np.median(best_scores))

logmodel.fit(x_train.iloc[:, best_chromosomes[-1]], y_train) # pick the last chromosome cause it's the last generation that has the best score
predictions = logmodel.predict(x_test.iloc[:, best_chromosomes[-1]]) # pandas will select those features that are set to True using iloc
print(f"\n▶ Accuracy after GA = {accuracy_score(y_test, predictions)}")
