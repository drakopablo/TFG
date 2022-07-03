#! /usr/bin/python
"""
@title: TFG -- AG con HV
@author: Pablo Guillén Marquina
"""
#############
## Imports ##
#############

import os
import re
import math
import time
import pygad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# Using Skicit-learn to split data into training and testing sets
# && Cross validation
# https://www.kaggle.com/code/jnikhilsai/cross-validation-with-linear-regression/notebook
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
	train_test_split, KFold, cross_val_score, GridSearchCV
	)
# Import the models we are using
from sklearn.ensemble import (
	RandomForestRegressor, RandomForestClassifier
	)
# RMSE
from sklearn.metrics import (
	mean_squared_error, accuracy_score
	)

############
## Clases ##
############

class Data:
	def __init__(self, filename, params = None):
		'''
			filename = fichero donde se encuentran los parámetros del AG
		'''
		# Diccionario con los parametros
		self.parameters = read_parameters(filename) if (params is None) else params
		# Guardamos el nombre del fichero (opcional) y el tipo de problema
		self.filename = self.parameters['filename']
		self.pt = self.parameters['pt']
		self.cor_col = self.parameters['cor_col']
		# Leemos el csv y lo almacenamos en un dataframe
		self.orig_df = pd.read_csv(self.filename)
		self.df = self.cured_df(self.orig_df, action=0) # Remove "missing values"

		# TODO: una vez tratados los datos, podemos calcular el modelo de regresión
		# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

		# Observed are the values we want to predict
		self.observed = np.array(self.df[self.cor_col])
		# Separamos las variables que vamos a utilizar de la que queremos predecir
		self.features = self.df.drop(self.cor_col, axis = 1)
		# Almacenamos los nombres de las variables
		self.feature_list = list(self.features.columns) # Todas menos la variable a evaluar
		# Convert to numpy array
		self.features = np.array(self.features)

		# Nº de atributos totales (sin el que validar)
		self.W = len(self.feature_list)
		# Calculamos con 0 atributos (features) escogidos (ZeroR)
		ZeroR = self.zero_rule_algorithm_regression if self.pt == 1 else self.zero_rule_algorithm_classification
		self.Z = ZeroR(self.observed) if self.pt == 1 else ZeroR(self.observed)
		# Imprimimos la Z o el Acc, segun caso
		if self.pt == 1:
			print("Z = " + str(self.Z))
		else:
			print("Acc = " + str(self.Z))

		# Calculamos los mejores parámetros a usar para los roa
		if self.parameters["best_params"]:
			self.best_parameters = self.best_params()
			print("Mejores parametros:\n", self.best_parameters)

	def getSolutionByBBDD(self, arff_filename):
		from scipy.io import arff
		# Extraemos los nombres de los atributos de la base de datos original
		all_col_names = self.df.columns.values
		# Para la solución no tendremos en cuenta el valor de 'cor_col'
		index_cor_col = np.where(self.cor_col == all_col_names)
		col_names = np.delete(all_col_names, index_cor_col)
		# Leemos el arff y lo almacenamos en un dataframe
		data = arff.loadarff(arff_filename)
		df = pd.DataFrame(data[0])
		# Leemos los attributos realmente usados por la bbdd
		used_col_names = df.columns.values
		# Formateamos el resultado a nuestra representación de la solución
		sol = [1 if (cn in used_col_names) else 0 for cn in col_names]
		# Devolvemos solución
		return sol

	def printSolutionFormat(self, solution):
		# Formateamos la solución
		sol = np.array(solution, dtype=str)
		sol = "[" + " ".join(sol) + "]"
		# Devolvemos solución
		return sol

	def best_params(self):
		st = datetime.now()
		# Grid de los parámetros (TODO: esto puede ir fuera y ser pasado como parámetro)
		param_grid = {
			'normalize': [True, False]
		}
		# Elegimos qué algoritmo con qué CV utilizamos para calcular los mejores parámetros
		random_seed = self.parameters["random_seed"]
		lm = LinearRegression()
		#print(LinearRegression().get_params().keys())
		cv = KFold(n_splits=self.parameters["kfolds"], random_state=random_seed, shuffle=True)
		# Realizamos un grid search
		CV_lm = GridSearchCV(estimator=lm, param_grid=param_grid, cv=cv)
		CV_lm.fit(self.features, self.observed)
		# Imprimimos el tiempo que ha tardado en procesarlo
		print("Ha tardado: "+"0"+str(datetime.now() - st)[:-3])
		# Devolvemos el resultado
		return CV_lm.best_params_

	def cured_df(self, df: pd.DataFrame, action : int = 0):
		'''

	    Parameters
	    ----------
	    df : pd.DataFrame
	        DESCRIPTION.
	    action : int
	        Indica qué acción realizar con los "missing values":
                0 : Elimina las filas con "missing values"
                1 : Calcula la media de los valores no "missing" y la asigna a los que sí
	    Returns
	    -------
	    df2 : TYPE
	        DESCRIPTION.

	    '''
		"""
		(*) "Curamos" la base de datos.
		En este caso podemos optar por eliminar las filas con valores nulos ya que solo son 4, lo que representa un (4/198) * 100 = 2,02% del total de datos.
		En el caso de que le asignasemos un valor nosotros, los cálculos que realizasemos con esos datos ya no serían iguales que teniendo los valores reales (si los hubiere).
		"""
		# Hacemos una copia del df
		df2 = df.copy()
		# Posibles valores nulos
		null_values = ['?', 'null']
		# Comprobamos si las columnas contienen valores nulos
		# Devuelve un objeto de tipo Series (panda)
		is_null = df2.isin(null_values).any()
		null_val = [is_null.index[i] for i in range(len(is_null)) if is_null[i] == True]
		#print("columnas con valores nulos: ", null_val,"\n")
		"""
		# Comprobamos las columnas con valores numéricos y no numéricos
		# Devuelve un objeto de tipo Series (panda)
		num_values = self.df2.applymap(lambda x: isinstance(x, (int, float))).all(0)
		#
		non_num = [i for i in range(len(num_values)) if num_values[i] == False]
		print(num_values[non_num])
		"""
		# Nº de filas de la base de datos
		n_rows = len(df2.index) #self.df2.shape[0]
		# Columnas de los datos que contienen valores nulos
		col_names = set()
		col_mean = dict()
		# Comprobamos qué filas tienen valores nulos
		serie = pd.Series([False] * n_rows)
		for i in range(len(null_val)):
			# Nombre de la columna
			name = null_val[i]
			set.add(col_names, name)
			# Comprobamos qué filas tienen valores nulos
			serie |= df2[name].isin(null_values)
			rows = df2.loc[serie]
			# TODO: Tratamiento para valores nulos. En nuestro caso: calcular la media
			if (action == 1):
				# Escogemos aquellas filas que no tengan valores nulos
				indexes = [i for i in range(0, len(df2[name])) if i not in list(rows.index)]
				values = list(df2[name])
				# Para esa columna, calculamos la media de los atributos no nulos (NA)
				mean = np.mean(np.array([values[i] for i in indexes], dtype=np.float32))
				col_mean[name] = str(mean)
				# Sustituimos los valores nulos por la media
				df2[name] = df2[name].replace(null_values,[str(mean)] * len(null_values))
		# TODO: Tratamiento para valores nulos. En nuestro caso:
		if (action == 0):
			# Guardamos (los indices de) las filas
			rows = df2.loc[serie]

			# Eliminamos las filas que contienen valores nulos
			df2 = df2.drop(rows.index)
		return df2

	# zero rule algorithm for regression
	def zero_rule_algorithm_regression(self, test):
		#https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/
		le = len(test)
		prediction = sum(test) / float(le)
		predicted = [prediction for i in range(le)]
    	# Calculamos el valor
		rmse = mean_squared_error(test, predicted, squared=False)
		return rmse

	# zero rule algorithm for classification
	# TODO: ¿Hay que buscar el accuracy para 0 atributos también?
	def zero_rule_algorithm_classification(self, test):
		#https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/
		le = len(test)
		prediction = max(set(test), key=list(test).count)
		predicted = [prediction for i in range(le)]
		# Calculamos el valor
		acc = accuracy_score(test, predicted)
		return acc
	###############
	## Fin clase ##
	###############

###############
## Funciones ##
###############

def getHVByBBDD(data, bbdd):
	# Leemos la solución
	solution = data.getSolutionByBBDD(bbdd)
	# Calculamos el nº de características seleccionadas
	N = list(solution).count(1)
	print("N =", N)
	# Calculamos RMSE
	RMSE = linealRegression(data, solution)
	print("RMSE =", RMSE)
	# Formateamos la solución
	print(data.printSolutionFormat(solution))
	# Calculamos el hipervolumen
	hv = fitness_function(data, solution)
	return hv

def read_parameters(filename="parameters.txt"):
	dicc = dict()
	f = open(filename, "r")
	lines = f.readlines()
	for line in lines:
		if len(line) > 1 and line.lstrip()[0] != "#":
			x = re.findall("[^=\n\r\s]+", line.strip())
			#print(x)
			# Comprobamos qué tipo de valor es (Caso default: string)
			value = x[1].replace("\"", "").replace("\'", "") # Quitamos "" y ''
			y = value.split(".")
			if (len(y) > 1 and y[0].isdigit()):
				# Caso de los reales
				value = float(value)
			elif (value.isdigit()):
				# Caso de los enteros
				value = int(value)
			elif (value[-1].isdigit()):
				# Caso de los enteros negativos (e.g. keep_parenting)
				value = -int(value[-1])
			elif value == 'True':
				# Caso True
				value = True
			elif value == 'False':
				# Caso False
				value = False
			# Asignamos la entrada al diccionario
			dicc[x[0]] = value
	# Cerramos el fichero
	f.close()
	# Devolvemos el diccionario
	return dicc

def rmse(train_features, train_observed, test_features, test_observed) -> float:
	scores = cross_val_score(estimator=lr,
							 X = train_features,
							 y = train_observed,
							 scoring = 'neg_root_mean_squared_error',
							 cv = cv,
							 n_jobs = -1, # Todos los cores disponibles
							 error_score = 'raise')
	# Calculate the squared mean absolute error
	rmse2 = np.mean(np.absolute(scores))
	return rmse2

def acc(train_features, train_observed, test_features, test_observed) -> float:
	# global lr
	# https://machinelearningknowledge.ai/keras-model-training-functions-fit-vs-fit_generator-vs-train_on_batch/#:~:text=Keras%20Train%20on%20batch%20%3A%20train_on_batch%20%28%29%20As,after%20that%2C%20the%20function%20updates%20the%20model%20parameters.
	# Calculate the accuracy
	lr_params = lr.get_params()
	lr2 = RandomForestClassifier(n_estimators=lr_params['n_estimators'],
                              random_state=lr_params['random_state'],
                              max_features=lr_params['max_features'],
                              max_depth=lr_params['max_depth']
                              )
	lr2.fit(train_features, train_observed)
    #lr2.train_batch()
	preds = lr2.predict(test_features)
	acc2 = float(accuracy_score(test_observed, preds))
	return acc2

def linealRegression(data, solution, test_size = 0.25) -> float:
	# Extraemos los indices de las columnas con valor = 0
	mfl = data.feature_list
	# Indices que queremos remover
	indices = [mfl[i] for i, x in enumerate(solution) if x == 0] + [cor_col]
	# Separamos las variables que vamos a utilizar de la que queremos predecir
	features = data.df.drop(indices, axis = 1)
	# Almacenamos los nombres de las variables (opcional)
	feature_list = list(features.columns) # Todas menos la variable a evaluar
	# Si no hay ningun feature escogido, por defecto ponemos "cor_col"
	if (len(feature_list) == 0):
		indices = [mfl[i] for i, x in enumerate(solution) if x == 0]
		features = data.df.drop(indices, axis = 1)
	# Convert to numpy array
	features = np.array(features)

	# Split the data into training and testing sets
	# TODO: ¿el train y el test de observed deberían ser el mismo para todo? Es decir, calcularlo solo una vez y leerlo de una variable global o similar
	train_features, test_features, train_observed, test_observed = train_test_split(features, data.observed, test_size = test_size, random_state = random_seed)

	# RMSE con CV
	# TODO: Creo que no se está calculando bien el accuracy para clasificación.
	# Porque en ningun momento se está usando el predict, solo el scores()
	# https://www.kaggle.com/code/sociopath00/random-forest-using-gridsearchcv/notebook
	roa = ROA(train_features, train_observed, test_features, test_observed)
	#roa = rmse_or_acc(features, data.observed, ST)
	return roa

def hv_rmse(Z, rmse_x, W, N):
	#hv = (Z - rmse_x) * (W - N)
	hv = (1 - rmse_x/Z) * (1 - N/W)
	return hv

def hv_acc(Z, acc_x, W, N):
	#hv = (acc_x) * (W - N)
	hv = (acc_x) * (1 - N/W)
	return hv

def hyper_volume(data, solution) -> float:
	# TODO: Esto es es lo que hay que cambiar en cada iteración del AG. Es decir, los "labels" son los mismos para todas las iteraciones, pero los "features" cambian según las variables escogidas por el algoritmo. Por ello,
	# TODO: se ha de recalcular los "features" para cada función fitness, calcular el valor predicho del modelo (lr), y calcular el hipervolumen (hv) correspondiente.
	# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
	roa = linealRegression(data, solution)
	N = list(solution).count(1)
	hv = HV(data.Z, roa, data.W, N)
	return (hv, N)


############################
## Algoritmo Genético - 1 ##
############################

# TODO: revisar
# https://stackoverflow.com/questions/69544556/passing-arguments-to-pygad-fitness-function
def fitness_func(solution, solution_idx):
	hv, N = hyper_volume(data, solution)
	aux = "\nNum variables = " + str(N) + "\nHV = " + str(hv) + "\nSolution:"+ str(solution) + "\n"
	if write:
		file_object.write(aux)
	return hv

def on_generation(solution):
	global count
	count += 1
	print("\tGeneración nº " + str(count) + " completada.")

def ag(cp, mp, num_genes : int = 1):
	global params, crossover_probability, mutation_probability
	# Parámetros del AG:
	fitness_function = fitness_func
	# Nº generaciones y padres
	num_generations = params['num_generations']
	num_parents_mating = params['sol_per_pop']#params['num_parents_mating']
	# Tamaño poblacion y genes
	sol_per_pop = params['sol_per_pop']
	num_genes = num_genes
	# Rango de las variables (binario [0,1])
	init_range_low = params['init_range_low']
	init_range_high = params['init_range_high']
	# Selección:
	parent_selection_type = "rank"#params['parent_selection_type'] # 'sss' funciona bien
	K_tournament = params['K_tournament']
	keep_parents = params['keep_parents']
	# Cruce:
	crossover_type = params['crossover_type']
	if not params['is_benchmark']:
		crossover_probability = params['crossover_probability']
	else:
		crossover_probability = cp
	# Mutación:
	mutation_type = params['mutation_type']
	if not params['is_benchmark']:
		mutation_probability = params['mutation_probability']
	else:
		mutation_probability = mp

	# Instancia del AG
	ga_instance = pygad.GA(# Características
			   num_generations=num_generations,
			   num_parents_mating=num_parents_mating,
			   sol_per_pop=sol_per_pop,
			   num_genes=num_genes,
			   # Rango
			   init_range_low=init_range_low,
			   init_range_high=init_range_high,
			   # Selección
			   parent_selection_type=parent_selection_type,
   			   K_tournament=K_tournament,
			   keep_parents=keep_parents,
			   # Cruce
			   crossover_type=crossover_type,
               crossover_probability = crossover_probability,
			   # Mutación
			   mutation_type=mutation_type,
			   mutation_probability = mutation_probability,
			   #mutation_percent_genes=mutation_percent_genes,
			   # Fitness
			   fitness_func=fitness_function,
			   # Representación binaria
			   gene_type=int,
			   gene_space=[0, 1],
			   # Otro parámetros
			   on_generation=on_generation,
			   suppress_warnings=True,  # La siguiente línea da warning
			   save_best_solutions=True # Guarda la mejor solucion de cada generación. (!) Puede producir desbordamiento de memoria
				)
	# Ejecutamos el algoritmo genético
	ga_instance.run()
	# Devolvemos la instancia
	return ga_instance


############################
## Algoritmo Genético - 2 ##
############################


def fitness_function(data, solution):
	hv, N = hyper_volume(data, solution)
	aux = "\nNum variables = " + str(N) + "\nHV = " + str(hv) + "\nSolution:"+ str(solution) + "\n"
	if write:
		file_object.write(aux)
	return hv

def generate_new_population(init_range_low, init_range_high, pop_size):
	#Creating the initial population. [low, high) <- para rango [0,1] : [low, high+1]
	new_population = np.random.randint(low=init_range_low,
										high=init_range_high+1,
										size=pop_size)
	return new_population

def reinitialize_population(pop, init_range_low, init_range_high, pop_size, old_pop_percent = 0.01):
	#Creating the initial population. [low, high) <- para rango [0,1] : [low, high+1]
	new_population = generate_new_population(init_range_low, init_range_high, pop_size)
	# The point at which crossover takes place between two population.
	crossover_point = np.uint8(math.ceil(pop.shape[0]*old_pop_percent))
	# The new population will have its first part from the previous population, while the rest from the new one
	new_population[0:crossover_point] = pop[0:crossover_point]
	return new_population

def reinitialize_population2(pop, init_range_low, init_range_high, extra_pop_size):
	#Creating the initial population. [low, high) <- para rango [0,1] : [low, high+1]
	new_population = generate_new_population(init_range_low, init_range_high, extra_pop_size)
	# The new population will have its first part from the previous population, while the rest from the new one
	new_population = pop + new_population
	return new_population

def cal_pop_fitness(data, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calcuates the sum of products between each input and its corresponding weight.
    fitness = [fitness_function(data, solution) for solution in pop]
    return fitness

# Calcula el fitness condicionalmente. Es decir, si el valor del hijo está en el padre, lo añade y no lo calcula de nuevo.
def cal_pop_fitness_cond(data, padres, fitness, hijos):
    '''
        padres_fit = lista de tuplas (padre, fitness)
        hijos = lista de individuos resultantes del AG en esa generación
    '''
    # Calculating the fitness value of each solution in the current population if it's not already calculated.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    fitness2 = [fitness[padres.index(solution)] if (solution in padres) else fitness_function(data, solution) for solution in hijos]
    return fitness2

# TODO: deprecated
def select_mating_pool(pop, fitness, num_parents):
	fitness2 = fitness.copy()
	parents = np.empty((num_parents, pop.shape[1]))
	for parent_num in range(num_parents):
	    max_fitness_idx = np.where(fitness2 == np.max(fitness2))
	    max_fitness_idx = max_fitness_idx[0][0]
	    parents[parent_num, :] = pop[max_fitness_idx, :]
	    fitness2[max_fitness_idx] = -99999999999
	return parents

def select_mating_pool2(pop, fitness, num_parents):
	parents = [(pop[i], fitness[i]) for i in range(num_parents)]
	parents = sorted(parents, key=lambda x: x[1], reverse=True)
	# De la población resultante, nos quedamos con los "num_parents" primeros (los mejores)
	#parents = np.array([parents[i][0] for i in range(num_parents)])
	parents = np.array(list(zip(*parents))[0])
	#print("aaaa\n", parents)# np.array(list(zip(*parents))[0]) )
	return parents

def crossover(parents, offspring_size, crossover_probability, err_margin = 0.00000001):
    # SINGLE POINT
    offspring = parents.copy() # np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)
    parents_shape_0 = parents.shape[0]
    for k in range(offspring_size[0]):
        # The condition to apply the crossover probability
        rprob = np.random.rand() + err_margin
        if (rprob <= crossover_probability):
            # Index of the first parent to mate.
            parent1_idx = k%parents_shape_0 #parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents_shape_0 #parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, mutation_probability, err_margin = 0.00000001):
	offspring_mutation = offspring_crossover.copy()
	# Mutation changes a single gene in each offspring randomly.
	for idx in range(offspring_mutation.shape[0]):
		# The condition to apply the mutation probability
		rprob = np.random.rand() + err_margin
		if (rprob <= mutation_probability):
		    random_value = np.random.randint(low=0, high=2, size = 1)
		    random_index = np.random.randint(low=0, high=offspring_mutation.shape[1], size = 1)
		    offspring_mutation[idx, random_index] = random_value
	return offspring_mutation

def np_matrix_to_list(np_matrix):
	aux = np_matrix.tolist()
	lista = [[int(x) for x in aux[y]] for y in range(len(aux))]
	return lista

def matrix_sol_fit(solutions, fitness, sol_per_pop, Z):
	# Convertimos la matriz de soluciones a una lista
	matrix = np_matrix_to_list(solutions)
	# Asociamos cada solución con su valor fitness
	matrix_fit = [(matrix[i], float(fitness[i])) for i in range(sol_per_pop)]
	# Ordenamos la solución por orden descendente
	matrix_sol_fit = sorted(matrix_fit, key=lambda x: x[1], reverse=True)
	# Nos quedamos con las primeras "sol_per_pop" soluciones ordenadas
	matrix_sol = np.array([matrix_sol_fit[i][0] for i in range(sol_per_pop)])
	return matrix_sol

def order_pop_by_fitness(pop, fitness, sol_per_pop):
	# Creamos una lista de tuplas que asocie la solucion con su fitness (para la nueva pop)
	pop_fit = [(pop[i], float(fitness[i])) for i in range(sol_per_pop)]
	# Calculamos la nueva pobablación y la ordenamos de forma descendente segun fitness
	pop2 = sorted(pop_fit, key=lambda x: x[1], reverse=True) # Z = ZeroR(fit)
	# De la población resultante, nos quedamos con los "sol_per_pop" primeros (los mejores)
	pop = np.array([pop2[i][0] for i in range(sol_per_pop)]) # np.array(pop2N[0:sol_per_pop][0], dtype=object)
	# Calculamos su fitness (sustituimos la variable anterior)
	fitness = np.array([pop2[i][1] for i in range(sol_per_pop)]) #np.array(pop2N[0:sol_per_pop][1], dtype=object)
	return pop, fitness

def ag2(cp, mp, qc = False, reset_pop = False):
	'''
        qc : Quick Convergence (Permite que la población tenga repeticiones de sus mejores padres/hijos, quitando diversidad)
            · False: NO queremos convergencia rápida (eliminamos duplicados)
            · True: SÍ queremos convergencia rápida (permitimos duplicados)
        reset_pop : Reset Population (Permite reinicializar la población si todos los individuos son el mismo. Solo funciona si qc = True)
            . False: NO queremos reinicializar la población
            . True: SÍ queremos reinicializar la población
	'''
	# https://www.kdnuggets.com/2018/07/genetic-algorithm-implementation-python.html
	# Parámetros del AG:
	# Nº generaciones y padres
	num_generations = params['num_generations']
	num_parents_mating = params['sol_per_pop']#params['num_parents_mating']
	# Tamaño poblacion y genes
	sol_per_pop = params['sol_per_pop']
	# Rango de las variables (binario [0,1])
	init_range_low = params['init_range_low']
	init_range_high = params['init_range_high']
	# Cruce:
	crossover_type = params['crossover_type'] # Default: Single Point
	if not params['is_benchmark']:
		crossover_probability = params['crossover_probability']
	else:
		crossover_probability = cp
	# Mutación:
	mutation_type = params['mutation_type'] # Default: Random
	if not params['is_benchmark']:
		mutation_probability = params['mutation_probability']
	else:
		mutation_probability = mp
	# Error margin
	err_margin = 0.00000001

	# Parte de la población antigua con la que nos quedamos (si QC y RP)
	old_pop_percent = 0.01

	# Algoritmo genético (GA):
	num_weights = data.features.shape[1] # Number of genes
	# The population will have "sol_per_pop" chromosome where each chromosome has "num_weights" genes
	pop_size = (sol_per_pop, num_weights)
	#Creating the initial population. [low, high) <- para rango [0,1] : [low, high+1]
	new_population = generate_new_population(init_range_low, init_range_high, pop_size)
	pop = new_population.copy()
	# Best outputs
	best_outputs = []
	# Calculamos la función fitness (sólo la 1a vez)
	fitness = cal_pop_fitness(data, pop)
    # Ordenamos la población
	pop, fitness = order_pop_by_fitness(pop, fitness, sol_per_pop)

	# For each generation
	for generation in range(num_generations):
		# Selecting the best parents in the population for mating.
		parents = select_mating_pool2(pop, fitness, num_parents_mating)

		# Generating next generation using crossover.
		offspring_crossover = crossover(parents, (sol_per_pop, num_weights), crossover_probability, err_margin)

		# Adding some variations to the offspring using mutation
		offspring_mutation = mutation(offspring_crossover, mutation_probability, err_margin)

		# (mu + lambda) Creating the new population based on the parents and offspring
		# Transformamos padres e hijos a listas
		padres = np_matrix_to_list(parents)
		hijos = np_matrix_to_list(offspring_mutation)
		# Creamos una lista de tuplas que asocie la solucion con su fitness (para padres e hijos)
		padres_fit = [(padres[i], float(fitness[i])) for i in range(sol_per_pop)] #list(zip(padres, fitness))
		fitness2 = cal_pop_fitness_cond(data, padres, fitness, hijos)
		hijos_fit = [(hijos[i], float(fitness2[i])) for i in range(sol_per_pop)] #list(zip(hijos, fitness2))
		# Calculamos la nueva pobablación y la ordenamos de forma descendente segun fitness
		pop2N = padres_fit + hijos_fit
		pop2N = sorted(pop2N, key=lambda x: x[1], reverse=True) # Z = ZeroR(fit)
		# Eliminamos individuos duplicados. De esta forma mantenemos la diversidad de la población, de otra forma podría suponer una convergencia prematura.
		# https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_population.htm
		# https://www.tutorialspoint.com/remove-duplicate-tuples-from-list-of-tuples-in-python#:~:text=Python%20Server%20Side%20Programming%20Programming%20When%20it%20is,If%20yes%2C%20it%20returns%20%27True%27%2C%20else%20returns%20%27False%27
		if not qc:
		    aux = [[a, b] for i, [a, b] in enumerate(pop2N) if not any(a == c for c, d in pop2N[:i])]
		    repeats = len(aux) - sol_per_pop # Comprobamos si tenemos al menos "sol_per_pop" individuos en la población
		    # Comprobamos que la población no sea menos de "sol_per_pop"
		    if (repeats >= 0):
		        pop2N = aux
		    # En caso de haber menos elementos:
		    elif (reset_pop):
			    # Nº de individuos que necesita la nueva población
			    extra_pop_size = -repeats
			    # Reinicio la población
			    print("Reinicializamos parte de la población. Necesitamos", extra_pop_size, " nuevos individuos")
			    # De la población resultante, nos quedamos con los "sol_per_pop" primeros (los mejores)
			    pop = np.array([aux[i][0] for i in range(len(aux))]) # np.array(pop2N[0:sol_per_pop][0], dtype=object)
			    # Reinicializamos parte de la población
			    pop2 = reinitialize_population2(pop, init_range_low, init_range_high, extra_pop_size)
			    # Calculamos la función fitness de nuevo
			    fitness2 = cal_pop_fitness_cond(data, pop, fitness, pop2)#cal_pop_fitness(data, pop2)
			    # Ordenamos la población
			    pop, fitness = order_pop_by_fitness(pop2, fitness2, sol_per_pop)
			    # Devolvemos la población
			    pop2N = [(pop[i], float(fitness[i])) for i in range(sol_per_pop)]
		    else:
			    # Si hay menos elementos, podemos repetir el mejor (o generar nuevos elementos)
			    aux.insert(0, (aux[0]*repeats))
			    pop2N = aux
				#pop2N = aux if (repeats == 0) else aux.insert(0, (aux[0]*repeats))
		# De la población resultante, nos quedamos con los "sol_per_pop" primeros (los mejores)
		pop = np.array([pop2N[i][0] for i in range(sol_per_pop)]) # np.array(pop2N[0:sol_per_pop][0], dtype=object)

		# Calculamos su fitness (sustituimos la variable anterior)
		fitness = np.array([pop2N[i][1] for i in range(sol_per_pop)]) #np.array(pop2N[0:sol_per_pop][1], dtype=object)
		# El mejor valor estará en la primera posición (ya que está ordenado descendientemente)
		max_fitness = fitness[0]
		# Lo añadimos a la lista de mejores soluciones
		best_outputs.append((list(pop)[0], max_fitness))

		# Terminamos la generacion
		print("\tGeneración nº " + str(generation+1) + " completada --- Max. Fitness = " + str(max_fitness))

		# TODO: reinicio de la población
		# Si todos los individuos son el mismo en la población, reinicializamos
		if (reset_pop and qc and ((fitness == max_fitness).sum() == sol_per_pop)):
		    print("Reinicializamos población, quedándonos con el", (old_pop_percent*100), "% de los mejores de la población anterior")
		    # Reinicializamos parte de la población
		    pop2 = reinitialize_population(pop, init_range_low, init_range_high, pop_size, old_pop_percent)
		    # Calculamos la función fitness de nuevo
		    fitness2 = cal_pop_fitness(data, pop2)
		    # Ordenamos la población
		    pop, fitness = order_pop_by_fitness(pop2, fitness2, sol_per_pop)

	# Recuperamos la mejor generacion donde está el mejor valor fitness
	# TODO: buscar dentro de las generaciones (bsg) aquella que tenga [máx HV, min Var]
	ngen = num_generations-1
	sols = [best_outputs[i][0] for i in range(num_generations)]
	bsf = [best_outputs[i][1] for i in range(num_generations)]
	bsg = bsf.index(max(bsf))
	# Devolvemos la instancia
	return (bsg, bsf, sols, ngen)

########################
## Mostrar resultados ##
########################


def saveAndShowFigure(bsf, bsg, ngen, precision = '%.3f', my_dpi=96, img_h=720, img_w=640, figure_filename="PyGAD_figure.jpg"):
	# https://stackoverflow.com/questions/66055330/how-can-i-save-a-matplotlib-plot-that-is-the-output-of-a-function-in-jupyter
	#https://queirozf.com/entries/add-labels-and-text-to-matplotlib-plots-annotation-examples
	start = 0
	plt.figure(figsize=(img_h/my_dpi, img_w/my_dpi), dpi=my_dpi)
	ys = bsf
	xs = [x for x in range(start, len(list(bsf)))]
	plt.plot(xs, ys,linestyle='--', marker='o', color='b', label='AG behaviour')
	# Comprobamos con last y actual los valores fitness ya impresos
	last = ""
	actual = ""
	# zip joins x and y coordinates in pairs
	for x,y in zip(xs,ys):
		# Comprobamos si ya hemos imprimido antes ese valor
		last = y
		if actual == "" or actual != last:
			# Si no se ha escrito antes, lo escribimos
			actual = last
			label = (precision % y)#"{:.2f}".format(y)
		elif actual == last:
			# Si ya se ha escrito antes, no lo escribimos
			label = ""
		# Para lo demás
		plt.annotate(label, # this is the text
			     (x,y), # these are the coordinates to position the label
			     textcoords="offset points", # how to position the text
			     xytext=(0,10), # distance from text to points (x,y)
			     ha='center') # horizontal alignment can be left, right or center
	# Marcamos la mejor solución
	plt.plot(bsg+start, bsf[bsg], marker='p', color='r', label='best solution')
	# Cambiamos la escala
	step = math.floor((ngen+1)*0.1)
	step = step if step > 0 else step+1
	stop = (ngen+1)+step if step > 1 else (ngen+1)
	rango = np.arange(start=start, stop=stop, step=step, dtype=int)
	plt.xticks([i for i in rango])#range(0,ngen+1)]) # [1, Nº generaciones + 1]
	# Mostramos los textos
	plt.legend()
	plt.title("Resultados AG")
	plt.xlabel("Nº de generaciones")
	plt.ylabel("Hipervolumen (HV)")
	# Guardamos la figura
	# TODO: podemos dar la opción de elegir el nombre del fichero a guardar y la extensión
	plt.savefig(figure_filename, dpi=my_dpi)
	# Mostramos el plot
	plt.show()

def showResults(ga_instance, num_genes, precision=6, my_dpi=96, img_h=720, img_w=640, figure=True, figure_filename="PyGAD_figure.jpg"):
	# Recuperamos la mejor generacion donde está el mejor valor fitness
	# TODO: buscar dentro de las generaciones (bsg) aquella que tenga [máx HV, min Var]
	bsg = ga_instance.best_solution_generation
	bsf = ga_instance.best_solutions_fitness
	sols = ga_instance.best_solutions
	ngen = ga_instance.num_generations

	# Mostramos la solución final
	solution = sols[bsg]
	solution_fitness = bsf[bsg]
	prediction = np.sum(np.array([1]*num_genes)*solution)

	# Guardamos los resultados en una variable para después poder guardarlo todo en un fichero
	res = "Number of needed variables: {prediction} variables\n".format(prediction=prediction)
	prec = ('%.'+str(precision)+'f')
	result = (prec % solution_fitness).rstrip('0').rstrip('.')
	res += "Fitness value of the best solution = {result}\n".format(result=result)
	res += "Parameters of the best solution:\n{solution}".format(solution=solution)
	print(res)

	# Mostramos la figura y la guardamos en un fichero JPG
	if figure:
		saveAndShowFigure(bsf, bsg, ngen, prec, my_dpi, img_h, img_w, figure_filename)
	return res, solution

def showResults2(bsg, bsf, sols, ngen, num_genes, precision=6, my_dpi=96, img_h=720, img_w=640, figure=True, figure_filename="PyGAD_figure.jpg"):
	# Mostramos la solución final
	#solution, solution_fitness, solution_idx = ga_instance.best_solution()
	solution = "[" + " ".join(str(x) for x in sols[bsg]) + "]"
	solution_fitness = bsf[bsg]
	prediction = np.sum(np.array([1]*num_genes)*sols[bsg])

	# Guardamos los resultados en una variable para después poder guardarlo todo en un fichero
	res = "Number of needed variables: {prediction} variables\n".format(prediction=prediction)
	prec = ('%.'+str(precision)+'f')
	result = (prec % solution_fitness).rstrip('0').rstrip('.')
	res += "Fitness value of the best solution = {result}\n".format(result=result)
	res += "Parameters of the best solution:\n{solution}".format(solution=solution)
	print(res)
	# Mostramos la figura y la guardamos en un fichero JPG
	if figure:
		saveAndShowFigure(bsf, bsg, ngen, prec, my_dpi, img_h, img_w, figure_filename)
	return res, solution

##########
## Main ##
##########

def main(filename, parameters = None, cp = 0.0, mp = 0.0):
	global data, params, pt, cor_col, random_seed, lr, cv, HV, ROA, write, count, file_object

	print("=================================================================")
    # Medimos el tiempo inicial:
	# Tiempo de inicio
	start_time = datetime.now()
	# Objeto que almacena los datos
	data = Data(filename=filename)
	# Tiempo transcurrido
	middle_time = (datetime.now() - start_time)
	print("-----------------------------------------------------------------")
	print("Tiempo en procesar los datos: " + "0"+str(middle_time)[:-3])
	print("-----------------------------------------------------------------")

    # Establecemos los parámetros:
	# Diccionario con los parámetros
	params = data.parameters# if (parameters is None) else parameters

	# Contador de las generaciones ya pasadas
	count = 0

	# Tipo de problema (0: Clasificación; 1: Regresión)
	pt = params['pt']
	# Nombre de la columna donde se encuentran los datos observados
	cor_col = params['cor_col']

	# Elegimos qué tipo de ag() ejecutar
	mu_lambda = params['mu_lambda']

    # Semilla de los random()
	random_seed = params['random_seed']

    # Cross validation:
	kfolds = params['kfolds']
	cv = KFold(n_splits=kfolds, random_state=random_seed, shuffle=True)

    # Lineal Regression Model:
    # Score Type
	ROA = rmse if pt == 1 else acc
	# Nº de árboles en el algoritmo MRL
	normalize = params['normalize']
	# Creamos la instancia del MRL
	lr = LinearRegression(normalize=normalize)

    # Hipervolumenes:
	HV = hv_rmse if pt == 1 else hv_acc

	# Variable auxiliar para decidir si imprimir o no los "print()"
	#verbose = params['verbose']
	# Variable auxiliar para decidir si escribir o no los resultados en un fichero
	write = params['write']
	# Variable auxiliar para decidir si mostrar y guardar o no la figura
	figure = params['figure']
	# Variable auxiliar donde se almacena el nombre y extensión de la figura
	figure_filename = params['figure_filename']

    # Mostrar resultados:
	if write:
    	# Fichero de salida
		output_file = params['output_file']
		# Open a file with access mode 'a'
		file_object = open(output_file, 'a')

	# Configuración del tamaño de la imagen de la figura generada
	my_dpi = params['my_dpi']# DPI del monitor
	img_h = params['img_h']
	img_w = params['img_w']

	# Precisión de la función fitness (nº de decimales)
	precision = params['precision']

	# Permitimos o no la "convergencia prematura" (quick convergence) del AG
	qc = params['qc']
	# Permitimos o no la "reinicialización de la población" (reset pop) del AG
	reset_pop = params['reset_pop']

	# Condición de benchmark
	#is_benchmark = params['is_benchmark']

    # Algoritmo Genético:
	print("[AG]:")
	middle_time = datetime.now()
	if mu_lambda == 1:
		bsg, bsf, sols, ngen = ag2(cp, mp, qc, reset_pop)
	else:
		ga_instance = ag(cp, mp, data.W)

    # Medimos el tiempo final:
	# Tiempo del algoritmo
	ag_time = (datetime.now() - middle_time)
	print("-----------------------------------------------------------------")
	print("Tiempo en ejecutar el algoritmo: "+"0"+str(ag_time)[:-3])
	print("-----------------------------------------------------------------")

    # Mostramos los resultados:
	#ga_instance.plot_fitness()
	if mu_lambda == 1:
		res, solution = showResults2(bsg, bsf, sols, ngen, data.W, precision, my_dpi, img_h, img_w, figure, figure_filename)
	else:
		res, solution = showResults(ga_instance, data.W, precision, my_dpi, img_h, img_w, figure, figure_filename)
	# Close the file
	if write:
		file_object.close()
	print("=================================================================")
	return res, solution

def check_performance(filename = 'parameters.txt', nruns : int = 1 , cp = 0.0, mp = 0.0, seed = None):
	global crossover_probability, mutation_probability, params
	params = read_parameters(filename)
	# Asignamos las probabilidades
	if params['is_benchmark']:
		crossover_probability = cp
		mutation_probability = mp
	else:
		crossover_probability = params["crossover_probability"]
		mutation_probability = params["mutation_probability"]
	crossover_probability = cp
	mutation_probability = mp
	# Tiempo de inicio
	start_time = datetime.now()
	# TODO: flag "isBenchmark" que imprime cp y mp en función de eso
	# Imprimimos los mensajes
	write_it_down = ""
	write_it_down += "------------------------------------------" + "\n"
	write_it_down += "Características para "+ str(nruns) + " runs" + "\n"
	if params['pt'] == 0:
		write_it_down += "Problema de Clasificación\n"
	elif params['pt'] == 1:
		write_it_down += "Problema de Regresión\n"
	write_it_down += "------------------------------------------" + "\n"
	write_it_down += "Selection type = " + str(params["parent_selection_type"]) + "\n"
	if params["parent_selection_type"] == 'tournament':
		write_it_down += "K Tournament = " + str(params["K_tournament"]) + "\n"
	write_it_down += "Number of generations = " + str(params["num_generations"]) + "\n"
	write_it_down += "Size of population = " + str(params["sol_per_pop"]) + "\n"
	write_it_down += "Number of parents kept = " + str(params["keep_parents"]) + "\n"
	write_it_down += "Crossover type = " + str(params["crossover_type"]) + "\n"
	if params['is_benchmark']:
		write_it_down += "Crossover probability = " + str(cp) + "\n"
	else:
		write_it_down += "Crossover probability = " + str(params["crossover_probability"]) + "\n"
	write_it_down += "Mutation type = " + str(params["mutation_type"]) + "\n"
	if params['is_benchmark']:
		write_it_down += "Mutation probability = " + str(mp) + "\n"
	else:
		write_it_down += "Mutation probability = " + str(params["mutation_probability"]) + "\n"
	write_it_down += "------------------------------------------" + "\n"
	print(write_it_down)

	# Ejecutamos los "runs"
	runs = []
	for i in range(1, (nruns+1)):
		# Establecemos Semilla
		if (seed is None):
			np.random.seed(i)
		else:
			np.random.seed(seed)
		res, solution = main(filename=filename, parameters=params, cp=cp, mp=mp)
		aux = re.findall("=.+", res.strip())
		fit = float(aux[0][1::].strip())
		runs.append(fit)
		print("results in iteration",i,"=",fit)
	# Calculamos los test estadísticos
	aux = "Results = {runs}\n".format(runs=runs)
	aux += "Min = {_min}\n".format(_min=np.min(np.array(runs)))
	aux += "Max = {_max}\n".format(_max=np.max(np.array(runs)))
	aux += "Mean = {mean}\n".format(mean=np.mean(np.array(runs)))
	aux += "Best solution = {sol}\n\n".format(sol = solution)
	print(aux)

	# Calculamos el tiempo
	end_time = (datetime.now() - start_time)
	print("Tiempo en ejecutar check_performance(): "+"0"+str(end_time)[:-3])
	# Devolvemos el resultado
	return str(write_it_down + aux)

def benchmark(filename = 'parameters.txt', nruns:int=1, D=1, filewriter = "resultados.txt", cp2 = 0, mp2 = 0, from_until = False):
	'''
		nruns = Nº de runs
		D = Nº de divisiones del en el rango [0,1]
        filewriter = Fichero donde escribir los resultados
        cp2 = Probabilidad de cruce (límite)
        mp2 = Probabilidad de mutación (límite)
        from_until:
            · False = from (cp2, mp2) to [max_cp2, max_mp2], both included
            · True = from [0.0, 0.0] to (cp2, mp2), both included
	'''
	# Tiempo de inicio
	start_time = datetime.now()

	# Open a file with access mode 'w'
	writer = open(filewriter, 'w')

	# Parámetros
	prec = 3		# Precision del redondeo (para evitar overflow)

	# Configuramos el benchmark
	start = 0.0		# Inicio del rango
	end = 1.0		# Final del rango (sin incluir)
	step = end / D	# Tamaño de la división
	end += step 	# Si queremos incluir el último valor, añadimos un "step"
	rango = np.arange(start, end, step).round(prec) # Redondeamos a "prec" decimales

	# (!) Las siguientes variables es para reanudar la ejecución por la última que nos hemos quedado
	#cp2 = 0.1
	#mp2 = 0.4
	run = from_until
	# Recorremos el rango de valores y ejecutamos el algoritmo
	for cp in rango:
		for mp in rango:
			if cp >= cp2 and mp >= mp2:
				run = not from_until
			if run:
				#print("[ cp =", cp, ", mp =", mp,"]")
				result = check_performance(filename, nruns, cp, mp)
				# Escribimos los resultados en el fichero
				writer.write(result)
	# Close the file
	writer.close()

	# Calculamos el tiempo y lo imprimimos
	end_time = (datetime.now() - start_time)
	print("Tiempo en ejecutar benchmark(): "+"0"+str(end_time)[:-3])

def benchmark2(filename = 'parameters.txt', nruns:int=1, filewriter = "resultados.txt", cp = 0, mp = 0):
	'''
		nruns = Nº de runs
        filewriter = Fichero donde escribir los resultados
        cp2 = Probabilidad de cruce
        mp2 = Probabilidad de mutación
	'''
	# Tiempo de inicio
	start_time = datetime.now()

	# Open a file with access mode 'w'
	writer = open(filewriter, 'w')

	# Ejecutamos el algoritmo
	result = check_performance(filename, nruns, cp, mp)
	# Escribimos los resultados en el fichero
	writer.write(result)
	# Close the file
	writer.close()

	# Calculamos el tiempo y lo imprimimos
	end_time = (datetime.now() - start_time)
	print("Tiempo en ejecutar benchmark(): "+"0"+str(end_time)[:-3])

def check_results(filereader='results.txt'):
	## Comprobamos el mejor algoritmo
	# Leemos el texto del fichero
	reader = open(filereader, 'r')
	lines = reader.readlines()
	text = "".join(str(line) for line in lines)
	# Aplicamos regex para encontrar los valores
	p = re.compile("Max .*")
	p2 = re.compile("Crossover probability .*")
	p3 = re.compile("Mutation probability .*")
	indexes = []
	values = []
	# Recorremos el texto
	for m in p.finditer(text):
		# Guardamos los valores
		indexes.append(int(m.start()))
		values.append(float(str(m.group())[6::].strip()))
	#print("Benchmark results:\n",indexes,"\n", values,"\n")
	# Comprobamos qué algoritmo ha conseguido el máximo valor
	length = 38    # Longitud de la cadena donde está el máximo (Max)
	max_value = max(values)
	max_index = values.index(max_value)
	# Leemos qué características han sido las mejores
	start_index = max_index-1 if max_index > 0 else 0
	i = indexes[start_index]+int(length)
	j = indexes[max_index]+int(length*3)
	text2 = (text[i:j]).split("\n", 1)[1]
	# Las imprimimos por pantalla
	print("Best results:\n" + text2)

###########
# Pruebas #
###########

# Ajuste de parámetros
def ajusteParametros(D, nruns):
	# Mu+lambda QC
	filename='parameters-ajuste-mu+lambda-qc.txt' #'parameters.txt'
	filewriter='results-ajuste-mu+lambda-qc.txt' #'results-prueba.txt'
	benchmark(filename=filename,nruns=nruns, D=D, filewriter=filewriter)
	# Mu+lambda NQC
	filename='parameters-ajuste-mu+lambda-nqc.txt' #'parameters.txt'
	filewriter='results-ajuste-mu+lambda-nqc.txt' #'results-prueba.txt'
	benchmark(filename=filename,nruns=nruns, D=D, filewriter=filewriter)
	# Mu+lambda QC RP
	filename='parameters-ajuste-mu+lambda-qc-rp.txt' #'parameters.txt'
	filewriter='results-ajuste-mu+lambda-qc-rp.txt' #'results-prueba.txt'
	benchmark(filename=filename,nruns=nruns, D=D, filewriter=filewriter)
	# Mu+lambda NQC RP
	filename='parameters-ajuste-mu+lambda-nqc-rp.txt' #'parameters.txt'
	filewriter='results-ajuste-mu+lambda-nqc-rp.txt' #'results-prueba.txt'
	benchmark(filename=filename,nruns=nruns, D=D, filewriter=filewriter)
	# Rank
	filename='parameters-ajuste-rank.txt' #'parameters.txt'
	filewriter='results-ajuste-rank.txt' #'results-prueba.txt'
	benchmark(filename=filename,nruns=nruns, D=D, filewriter=filewriter)
	# Tournament
	filename='parameters-ajuste-tournament.txt' #'parameters.txt'
	filewriter='results-ajuste-tournament.txt' #'results-prueba.txt'
	benchmark(filename=filename,nruns=nruns, D=D, filewriter=filewriter)
	# RWS
	filename='parameters-ajuste-rws.txt' #'parameters.txt'
	filewriter='results-ajuste-rws.txt' #'results-prueba.txt'
	benchmark(filename=filename,nruns=nruns, D=D, filewriter=filewriter)

def ejecucionesLargas(nruns2):
	# Mu+lambda QC
	filename='parameters-inter-mu+lambda-qc.txt' #'parameters.txt'
	filewriter='results-inter-mu+lambda-qc.txt' #'results-prueba.txt'
	benchmark2(filename=filename,nruns=nruns2, filewriter=filewriter)
	# Mu+lambda NQC
	filename='parameters-inter-mu+lambda-nqc.txt' #'parameters.txt'
	filewriter='results-inter-mu+lambda-nqc.txt' #'results-prueba.txt'
	benchmark2(filename=filename,nruns=nruns2, filewriter=filewriter)
	# Mu+lambda QC RP
	filename='parameters-inter-mu+lambda-qc-rp.txt' #'parameters.txt'
	filewriter='results-inter-mu+lambda-qc-rp.txt' #'results-prueba.txt'
	benchmark2(filename=filename,nruns=nruns2, filewriter=filewriter)
	# Mu+lambda NQC RP
	filename='parameters-inter-mu+lambda-nqc-rp.txt' #'parameters.txt'
	filewriter='results-inter-mu+lambda-nqc-rp.txt' #'results-prueba.txt'
	benchmark2(filename=filename,nruns=nruns2, filewriter=filewriter)
	# Rank
	filename='parameters-inter-rank.txt' #'parameters.txt'
	filewriter='results-inter-rank.txt' #'results-prueba.txt'
	benchmark2(filename=filename,nruns=nruns2, filewriter=filewriter)
	# Tournament
	filename='parameters-inter-tournament.txt' #'parameters.txt'
	filewriter='results-inter-tournament.txt' #'results-prueba.txt'
	benchmark2(filename=filename,nruns=nruns2, filewriter=filewriter)
	# RWS
	filename='parameters-inter-rws.txt' #'parameters.txt'
	filewriter='results-inter-rws.txt' #'results-prueba.txt'
	benchmark2(filename=filename,nruns=nruns2, filewriter=filewriter)

def pruebaSuelta():
	# Open a file with access mode 'w'
	writer = open(filewriter, 'w')
	result = check_performance(filename=filename, nruns=1, seed=12345)
	# Escribimos los resultados en el fichero
	writer.write(result)
	# Close the file
	writer.close()

def calcularHVsBBDD():
	# Objeto que almacena los datos
	main(filename='parameters-bbdd.txt')
	print("Hipervolumenes BBDDs:")
	# ORIGINAL
	bbdd = "wpbc-regression-no-missing-values.arff"
	hv = getHVByBBDD(data, bbdd)
	print("ORIGINAL:",hv,"\n")
	# MU+LAMBDA NQC
	bbdd = "wpbc-regression-no-missing-values-mulambda-nqc.arff"
	hv = getHVByBBDD(data, bbdd)
	print("MU+LAMBDA-NQC:",hv,"\n")
	# BEST FIRST
	bbdd = "wpbc-regression-no-missing-values-best-first.arff"
	hv = getHVByBBDD(data, bbdd)
	print("BEST FIRST:",hv,"\n")
	# ANT
	bbdd = "wpbc-regression-no-missing-values-ANT.arff"
	hv = getHVByBBDD(data, bbdd)
	print("ANT:",hv,"\n")
	# CUCKOO
	bbdd = "wpbc-regression-no-missing-values-CUCKOO.arff"
	hv = getHVByBBDD(data, bbdd)
	print("CUCKOO:",hv,"\n")
	# PSO
	bbdd = "wpbc-regression-no-missing-values-PSO.arff"
	hv = getHVByBBDD(data, bbdd)
	print("PSO:",hv,"\n")
	# FIN


########
# Main #
########

if __name__ == "__main__":
	# Si da problemas cambiamos el directorio de trabajo a aquel donde se encuentran nuestros ficheros
	working_dir = 'C:/Users/Dragg/Desktop/Desktop/Ing_Informatica/4º/TFG'
	os.chdir(working_dir)
	# Ejecutamos el main
	filename='parameters-bbdd.txt' #'parameters.txt'
	filewriter='erase.txt' #'results-prueba.txt'
	#benchmark2(filename=filename,nruns=1, filewriter=filewriter)
	#params = read_parameters(filename)
	#main(filename=filename)
	#print(check_performance(10, 0.0, 0.0))
	#benchmark(filename=filename,nruns=10, D=10, filewriter=filewriter)#, cp2=0.4, mp2=0.5)
	#check_results(filewriter)
	D = 10
	nruns = 10
	nruns2 = 30
	#ajusteParametros(D=D, nruns=nruns)
	#ejecucionesLargas(nruns2=nruns2)
	#check_results(filereader="results-ajuste-rws"+".txt")
	#pruebaSuelta()
	calcularHVsBBDD()

# Lo siguiente son cosas aparte. No hacer caso
""" # To Excel
def metodo():
    # TODO: leo los resultados de la media, min y max y se los paso a este metodo
    aux = ""
    # aux son todos los resultados
    aux2 = aux.split(sep=",")
    step = 11
    aux3 = np.empty([11, 11], dtype=np.float64)
    j = -1
    # Metemos los valores en una matriz por columnas
    for i, value in enumerate(aux2):
        if (i%step == 0):
            j+=1
        aux3[i%step, j] = value
    # Imprimimos con formato bueno
    for k in range(step):
        print("\nCrossover_prob 0."+ str(k) + ":")
        for elem in aux3[k].astype(str):
            print(elem.replace(".", ","))
"""

""" # Tiempo
def timesN(n):
  return lambda a : a * n

def tiempoT(t):
    dd = 0
    hh = 0
    mm = 0
    ss = sum(t)
    if (ss > 60):
        mm = int(ss/60)
        ss = int(ss%60)
        if (mm > 60):
            hh = int(mm/60)
            mm = int(mm%60)
            if (hh > 24):
                dd = int(hh/60)
                hh = int(hh%60)

    aux = "{dd}d {hh}h {mm}\' {ss}\'\'\n".format(dd=dd, hh=hh, mm=mm, ss=ss)
    return aux

times10 = timesN(10)
times30 = timesN(30)
timesT = timesN(1210)

# T es lo que cambia
#t = [12] + [20]*3 + [32]*3
t = [1] + [2]*3 + [3.7]*3

t10 = [times10(i) for i in t]
t30 = [times30(i) for i in t]
tT = [timesT(i) for i in t]

print("Tiempo t --", tiempoT(t))
print("Tiempo t10 --",tiempoT(t10))
print("Tiempo t30 --",tiempoT(t30))
print("Tiempo tT --",tiempoT(tT))
"""
