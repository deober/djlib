import numpy as np
import djlib.clex.clex as cl
import thermocore.geometry.hull as thull
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLarsCV


class GeneticAlgorithm:
    def __init__(
        self,
        target_values,
        composition,
        feature_matrix,
        population_size,
        mutation_rate,
        crossover_rate,
        selection_function,
        crossover_function,
        mutation_function,
        regularization_function=None,
        initial_chromosome=None,
        weighted_feature_matrix=None,
        weighted_target_values=None,
    ):
        self.target_values = target_values
        self.weighted_target_values = weighted_target_values
        self.composition = composition
        self.feature_matrix = feature_matrix
        self.weighted_feature_matrix = weighted_feature_matrix
        self.population_size = population_size
        self.chromosome_size = feature_matrix.shape[1]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        if regularization_function is None:
            self.regularization_function = LassoLarsCV(
                fit_intercept=False, n_jobs=-1, max_iter=50000
            )
        else:
            self.regularization_function = regularization_function
        self.hall_of_fame = []
        self.initial_chromosome = initial_chromosome

    def run(self, generations):

        population = generate_population(self.population_size, self.chromosome_size)
        population[:, 0:3] = np.ones(population.shape[0]).reshape(-1, 1)

        if self.initial_chromosome is not None:
            population[0, :] = self.initial_chromosome
        for iteration in range(generations):
            print(f"Generation {iteration}")

            # Calculate fitness for each member
            fitness = self._fitness_function(population)

            # Check if the top member of the population is better than the best member of the hall of fame.
            # If so, add it to the hall of fame.
            # If hall of fame is empty, add the top member of the population.
            if len(self.hall_of_fame) == 0:
                self.hall_of_fame.append(
                    {
                        "chromosome": population[np.argmax(fitness), :],
                        "fitness": np.max(fitness),
                    }
                )
            elif np.max(fitness) > self.hall_of_fame[-1]["fitness"]:
                self.hall_of_fame.append(
                    {
                        "chromosome": population[np.argmax(fitness), :],
                        "fitness": np.max(fitness),
                    }
                )

            # Select survivors
            population = self.selection_function(population, fitness)

            # Perform crossover on the population
            population = self.crossover_function(population, fitness)
            population[:, 0:3] = np.ones(population.shape[0]).reshape(-1, 1)

            # mutate the population
            population = self.mutation_function(population, self.mutation_rate)
            population[:, 0:3] = np.ones(population.shape[0]).reshape(-1, 1)

        # Calculate the final fitness, and sort the population by fitness (descending)
        fitness = self._fitness_function(population)
        population = population[np.argsort(fitness)[::-1], :]
        return population, self._fitness_function(population)

    def _fitness_function(self, population) -> np.ndarray:
        """Wrapper around model_fitness function. 
        Takes a population of chromosomes, and returns the fitness values corresponding to each member of the population.

        Parameters
        ----------
        population : array-like
            The current population of chromosomes.

        Returns
        -------
        array-like
            The fitness values corresponding to the population members.
        """
        fitness = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            # Downsample features according to the chromosome. Then fit.
            selected_features = np.where(population[i, :] == 1)[0]

            # If a weighted feature matrix is provided, use it in place of the feature matrix when fitting eci.
            if (
                self.weighted_feature_matrix is not None
                and self.weighted_target_values is not None
            ):
                eci = self.regularization_function.fit(
                    self.weighted_feature_matrix[:, selected_features],
                    self.weighted_target_values,
                ).coef_
                predicted_energies = (
                    self.weighted_feature_matrix[:, selected_features] @ eci
                )
            elif (
                self.weighted_feature_matrix is None
                and self.weighted_target_values is None
            ):
                eci = self.regularization_function.fit(
                    self.feature_matrix[:, selected_features], self.target_values
                ).coef_
            else:
                raise ValueError(
                    "Must provide both a weighted feature matrix and a weighted target values array."
                )
            predicted_energies = self.feature_matrix[:, selected_features] @ eci
            eci = self.regularization_function.fit(
                self.feature_matrix[:, selected_features], self.target_values
            ).coef_
            predicted_energies = self.feature_matrix[:, selected_features] @ eci

            fitness[i] = model_fitness(
                true_comp=self.composition,
                true_energies=self.target_values[0 : self.composition.shape[0]],
                predicted_comp=self.composition,
                predicted_energies=predicted_energies[0 : self.composition.shape[0]],
                complexity=np.count_nonzero(population[i, :])
                / population[i, :].shape[0],
            ) * (
                (population[i, :].shape[0] - np.count_nonzero(population[i, :]))
                / population[i, :].shape[0]
            )
        return fitness


def generate_population(population_size, chromosome_size) -> np.ndarray:
    """Creates a population of random binary chromosomes.
    Ensures that all members of the population have a 1 in the first column, to allow a constant offset term.

    Parameters
    ----------
    population_size : int
        The number of members (chromosomes) in the population.
    chromosome_size : int
        The number of genes in each chromosome.
    """
    population = np.random.randint(0, 2, (population_size, chromosome_size))
    # ensure that the first column is all 1s.
    population[:, 0] = np.ones(population_size)
    return population


def bit_flip_function(population: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Mutates a chromosome by randomly flipping bits.
    The mutation rate is the probability that each bit will be flipped.

    Parameters
    ----------
    chromosome : array-like
        The chromosome to be mutated.
    mutation_rate : float
        The probability that each bit will be flipped.

    Returns
    -------
    array-like
        The mutated chromosome.
    """
    for chromosome in population:
        for i in range(chromosome.shape[0]):
            if np.random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]
    return population


def crossover_function(population, fitness) -> np.ndarray:
    """Performs crossover on a population. Crosses the top member of the population with all other members. 
    
    Parameters
    ----------
    population : array-like
        The current population of chromosomes.
    fitness : array-like
        The fitness values corresponding to each member of the population.
    
    Returns
    -------
    array-like
        The new population after crossover.
    """
    new_population = []

    # Sort the population by fitness, from high to low fitness
    population = population[np.argsort(fitness)[::-1], :]

    for i in range(population.shape[0]):
        if i == 1:
            new_population.append(population[i, :])
        else:
            # Mix the top member with the current member. Each bit has a 50% chance of being from the top member.
            new_population.append(
                np.where(
                    np.random.random(population.shape[1]) < 0.5,
                    population[0, :],
                    population[i, :],
                )
            )

    return np.array(new_population)


def model_fitness(
    true_comp: np.ndarray,
    true_energies: np.ndarray,
    predicted_comp: np.ndarray,
    predicted_energies: np.ndarray,
    complexity: np.ndarray,
):
    """Determines the performance of a given model prediction. Takes composition and energy arrays for the 
        true and predicted values, and returns a single value [0,1] representing the performance of the model.
    
    Parameters
    ----------
    true_comp : array-like
        Compositions corresponding to the values in true_energies.
    true_energies : array-like
        Energies predicted by a high accuracy Hamiltonian, used as target values. 
    predicted_comp : array-like
        Compositions corresponding to the values in predicted_energies.
    predicted_energies : array-like
        Predicted energies from a surrogate model. 
    """

    # Find the "true ground states"
    true_hull = thull.full_hull(true_comp, true_energies)
    true_ground_states, _ = thull.lower_hull(true_hull)

    # Compute the gsa metric from djlib.clex.clex
    holistic_gsa = cl.ground_state_accuracy_metric(
        true_ground_state_indices=true_ground_states,
        composition_predicted=true_comp,
        energy_predicted=predicted_energies,
    )

    # Compute the DFT chemical potential window ground state accuracy metric
    dft_gsa = cl.gsa_fraction_correct_DFT_mu_window_binary(
        predicted_comp=predicted_comp,
        predicted_energies=predicted_energies,
        true_comp=true_comp,
        true_energies=true_energies,
    )

    # Compute the rmse.
    rmse = np.power(mean_squared_error(true_energies, predicted_energies), (1 / 2))

    # relevant rmse scales are between [0,~3e-2] of eV per primitive cell.
    # Normalize the rmse to this range.
    # If it is above 1, set it to 1.
    scaled_rmse = rmse / 0.03
    if scaled_rmse > 1:
        scaled_rmse = 1

    # Compute a fitness value that takes into account all three metrics.
    # The two gsa metrics are best when the are high (~1)
    # The rmse is best when it is low (~0).

    return holistic_gsa * dft_gsa * (1 - scaled_rmse) * (1 - complexity)


def select_survivors(population, fitness):
    """Takes the current population, and the corresponding fitness values. 
    Ranks the population members. Preserves the top 50% of the population, and replaces the rest with new random members.

    Parameters
    ----------
    population : array-like
        The current population of chromosomes.
    fitness : array-like
        The fitness values corresponding to the population members.
    
    Returns
    -------
    array-like
        The new population.
    """

    # Rank the population members based on their fitness.
    ranked_population = population[np.argsort(fitness)]

    # Generate new members
    new_population = generate_population(population.shape[0], population.shape[1])

    # Preserve the top 50% of the population.
    new_population[: int(len(ranked_population) / 2)] = ranked_population[
        : int(len(ranked_population) / 2)
    ]

    return new_population

