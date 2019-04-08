import math
import operator
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.ion()
plt.figure(figsize=(10, 5))


def plotTSP(generation, path, points, path_distance, save, num_iters=1):
    """
    generation: The generation number to display
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    path_distance: the distance to display in the figure
    save: True if saving to final_route.png, False otherwise
    num_iters: number of paths that are in the path list

    SOURCE: https://gist.github.com/payoung/6087046
    """
    ### MOD: Brian Bennett

    plt.clf()
    plt.suptitle("Tennessee Traveling Postal Worker - Generation " + str(generation) + \
                 "\nPath Length: " + str(path_distance))
    ### END MOD

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    x = [];
    y = []
    for i in path:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, 'ko')

    # Set a scale for the arrow heads (there should be a reasonable default for this)
    a_scale = float(max(x)) / float(2500)  # MOD: Brian Bennett
    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = [];
            yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                      head_width=a_scale, color='r',
                      length_includes_head=True, ls='dashed',
                      width=0.001 / float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i + 1] - xi[i]), (yi[i + 1] - yi[i]),
                          head_width=a_scale, color='r', length_includes_head=True,
                          ls='dashed', width=0.001 / float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=a_scale,
              color='b', length_includes_head=True)
    for i in range(0, len(x) - 1):
        plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]), head_width=a_scale,
                  color='b', length_includes_head=True)

    if save:
        plt.savefig("final_route.png")

    plt.pause(1)


class GeneticSearch:
    """
        Class: GeneticSearch
    """

    def __init__(self, origin,
                 points,
                 cities,
                 generations,
                 population_size,
                 mutation_rate,
                 elite_size=0.05,
                 sp_factor=2
                 ):
        """

        :param origin:
        :param generations:
        :param points:
        :param cities:
        :param population_size:
        :param mutation_rate:
        :param sp_factor: the sort population factor
        """
        self.population = None
        self.points = points
        self.cities = cities
        self.chromosome_size = len(self.points)
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.origin = origin
        self.origin_index = self.points.index(self.origin)
        self.values = []
        self.sp_factor = sp_factor
        self.elite_size = self.get_int_value(self.population_size, elite_size)

    def print_population(self, generation, chromosomes):
        index = 0
        print("===== GENERATION %d" % generation)
        for chromosome in self.population:
            print("Index %5d , Fitness %0.4f : %s" % (index, chromosome[1], ''.join(str(chromosome[0]))))
            index = index + 1
            if index > chromosomes:
                break

    def initialize_population(self):

        self.population = []
        _population = []
        init_individual = [x for x in range(self.chromosome_size)]
        # TODO: DONE::: This code generates a random initial population.
        #       You may adjust this code in any way that you believe would help.]
        larger_population = self.get_int_value(self.population_size, self.sp_factor)
        for i in range(larger_population):
            fitness, individual = self.create_individual(init_individual)

            # Prevent duplicate individuals in the initial population
            while [individual, fitness] in _population:
                fitness, individual = self.create_individual(init_individual)

            # POPULATION NODES are in the form [chromosome, fitness]
            _population.append([individual, fitness])

        _population.sort(key=operator.itemgetter(1), reverse=True)

        # add just the top self.population_size to population and delete excess
        self.population = _population[:self.population_size]
        del _population

    def create_individual(self, init_individual):
        individual = init_individual[:]
        random.shuffle(individual)
        individual = self.put_start_city_first(individual)
        fitness = self.fitnessfcn(individual)
        return fitness, individual

    def put_start_city_first(self, individual):
        individual.remove(self.origin_index)
        individual = [self.origin_index] + individual
        return individual

    @staticmethod
    def straight_line_distance(p1, p2):
        """
            Return the Euclidian Distance between p1 and p2
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def route_distance(self, individual):
        """
            Determine the distance for the entire route
        """
        distance = 0

        tour = individual + [self.origin_index]
        p2 = None
        index = 0
        while p2 != self.origin:
            p1 = self.points[tour[index]]
            p2 = self.points[tour[index + 1]]
            distance += self.straight_line_distance(p1, p2)
            index += 1

        return distance

    def fitnessfcn(self, individual):
        """
            Return the negative route distance so it can be maximized.
        """
        return -self.route_distance(individual)

    def select_parents(self, reproduction_pool):
        """
            Selects two parents from the population
        """
        # TODO: Consider a selection strategy
        parent1, parent2 = sorted(random.sample(reproduction_pool,2))

        return parent1[0], parent2[0]

    def reproduce2(self, reproduction_pool):
        parent1, parent2 = self.select_parents(reproduction_pool)
        midpoint = random.randint(1, self.chromosome_size - 1)

        child1 = parent1[midpoint:]
        child2 = parent2[:midpoint]

        add_c1 = [x for x in parent2 if x not in child1]
        add_c2 = [x for x in parent1 if x not in child2]

        return child1 + add_c1, child2 + add_c2

    def reproduce(self, reproduction_pool):
        """
            Reproduce using parent1 and parent2 and a crossover
             strategy.
        """

        # TODO: Implement a reproduction (e.g., crossover) strategy
        children = []
        for _ in range(2):
            parent1, parent2 = self.select_parents(reproduction_pool)
            city_1 = int(random.random() * len(parent1))
            city_2 = int(random.random() * len(parent1))
            while city_1 == city_2:
                city_2 = int(random.random() * len(parent1))

            start_city, end_city = min(city_1, city_2), max(city_1, city_2)
            p1 = [parent1[i] for i in range(start_city, end_city)]
            p2 = [city for city in parent2 if city not in p1]
            child = p1 + p2
            children.append(self.put_start_city_first(child))

        return children[0], children[1]

    def mutate(self, child):
        """
            Mutation Strategy
        """
        # TODO: Implement a mutation strategy
        for i in range(len(child)):
            if random.random() < self.mutation_rate:
                swapper = self.get_int_value(len(child) - 1, random.random())
                child[i], child[swapper] = child[swapper], child[i]
        return self.put_start_city_first(child)

    def print_result(self):
        """
            Displays the resulting route in the console.
        """
        individual = self.population[0][0]
        fitness = self.population[0][1]

        print(" Final Route in %d Generations" % self.generations)
        print(" Final Distance : %5.3f\n" % -fitness)

        counter = 1

        for index in individual:
            print("%2d. %s" % (counter, self.cities[index]))
            counter += 1

        print("%2d. %s" % (counter, self.cities[self.origin_index]))

    def run(self):
        """
            Run the genetic algorithm. Note that this method initializes the
             first population.
        """
        generations = 0

        # TODO: Update Initialization
        self.initialize_population()
        while generations <= self.generations:

            # get population for this generation
            reproduction_pool: list = self.elite_tournament_selectn()
            # using the elites to start new population
            # new_population = reproduction_pool[:]
            new_population = self.population[:self.elite_size]

            while len(new_population) < self.population_size:

                # TODO: Update reproduction

                child1, child2 = self.reproduce(reproduction_pool)
                # child1, child2 = self.reproduce2(reproduction_pool)

                # TODO: Update Mutation
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2)

                fitness1 = self.fitnessfcn(child1)
                fitness2 = self.fitnessfcn(child2)

                new_population.append([child1, fitness1])
                new_population.append([child2, fitness2])

            generations = generations + 1
            new_population.sort(key=operator.itemgetter(1), reverse=True)

            self.population = new_population
            self.values.append(self.population[0][1])
            # TODO: Change display rate as needed. Set by 1000 as default.
            if generations % 1000 == 0 or generations >= self.generations:
                print("Generation: %d" % generations, "Fitness: %f" % self.population[0][1])
                if generations == self.generations:
                    plotTSP(generations, self.population[0][0], self.points, self.population[0][1], True)
                # else:
                #     plotTSP(generations, self.population[0][0], self.points, self.population[0][1], False)

                plt.figure("Genetic Search - Best Fitness by Generation")
                plt.plot(self.values)
                plt.show()
        self.print_result()

    def elite_tournament_selectn(self):
        chosen = []
        percentage_of_participants = 0.05
        participants = self.get_int_value(len(self.population), percentage_of_participants)
        for _ in range(self.elite_size):
            aspirants = []
            for _ in range(participants):
                aspirant = random.choice(self.population)
                while aspirant in chosen:
                    aspirant = random.choice(self.population)
                aspirants.append(aspirant)
            chosen.append(max(aspirants, key=operator.itemgetter(1)))
        return chosen

    @staticmethod
    def get_int_value(quantity, percentage):
        return math.ceil(quantity * percentage)


def get_parameters_grid():
    args = [
        [random.randint(*GENERATION_RANGE) for _ in range(NUM_ITERATIONS)],
        [random.randint(*POPULATION_SIZE_RANGE) for _ in range(NUM_ITERATIONS)],
        [round(random.uniform(*MUTATION_RATE_RANGE), 2) for _ in range(NUM_ITERATIONS)],
        [round(random.uniform(*ELITE_SIZE), 2) for _ in range(NUM_ITERATIONS)],
    ]
    return list(map(list, zip(*args)))


if __name__ == '__main__':

    GENERATION_RANGE = 1000, 1500
    POPULATION_SIZE_RANGE = 200, 400
    MUTATION_RATE_RANGE = 0.1, 0.15
    ELITE_SIZE = 0.01, 0.1
    NUM_ITERATIONS = 15

    po_coordinates = "coordinates.txt"
    post_office_names = "post_offices.txt"
    start_office = "Johnson City Post Office, TN"
    locations = list(np.loadtxt(po_coordinates, delimiter=','))
    cities = [line.rstrip('\n') for line in open(post_office_names)]
    points = []
    paths = []
    start_office_index = [i for i in range(len(cities)) if cities[i] == start_office][0]

    loc_x = [x for x, y in locations]
    loc_y = [y for x, y in locations]
    loc_c = ["black" for _ in range(len(locations))]

    for i in range(0, len(loc_x)):
        points.append((loc_x[i], loc_y[i]))

    # origin, generations, points, population_size, mutation_rate
    origin = (locations[start_office_index][0], locations[start_office_index][1])

    # TODO: Adjust parameters as needed
    # Parameters: 1. origin location,
    #             2. number of generations,
    #             3. locations as a list of tuples,
    #             4. list of city names,
    #             5. number of individuals in each generation,
    #             6. mutation rate
    gs = GeneticSearch(origin, points, cities, 1370, 1000, 0.25, 0.05, 2)
    gs.run()
    gs = GeneticSearch(origin, points, cities,1370, 400, 0.25, 0.05, 2)
    gs.run()

    gs = GeneticSearch(origin, points, cities, 1261, 400, 0.11, 0.11, 2)
    gs.run()
    gs = GeneticSearch(origin, points, cities, 1370, 400, 0.25, 0.05, 2)
    gs.run()

    gs = GeneticSearch(origin, points, cities, 1484, 427, 0.13, 0.04, 2)
    gs.run()

    param_results = []
    params = []
    params.extend(get_parameters_grid())
    #
    for args in params:
        gs = GeneticSearch(origin, points, cities, *args)
        gs.run()
        args.append(sorted(gs.values, reverse=True)[0])
        param_results.append(args)
        df = pd.DataFrame(param_results,
                          columns=['generation', 'population', 'mutation_rate', 'elite_percent', 'fitness'])
        print(df.sort_values('fitness', ascending=False), end="\r")

    # plt.close()

# 1370         382           0.19 -6.285226
# gs = GeneticSearch(origin, points, cities,1370, 382, 0.25, 2)
# 0 1261, 400, 0.11, 0.11 - 5.940379
#
