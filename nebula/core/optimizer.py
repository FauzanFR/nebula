"""
GeneticOptimizer

Implements genetic algorithm for hyperparameter search with:
- Sobol sequence initialization
- Hybrid parent selection (tournament + chaotic)
- Adaptive mutation strategies
- Elite preservation
"""

import random

import numpy as np

class GeneticOptimizer:
    def __init__(self, param_space, population_size=50, elite_size=5, mutation_rate=0.2):
        """
        Initialize the evolutionary optimizer.
        
        Args:
            param_space      : Parameter search space definition
            population_size : Number of candidates per generation
            elite_size      : Top performers preserved between generations  
            mutation_rate   : Probability of parameter mutation
        """
        self.param_space = param_space
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.history = []
        self.sobol_state = 0
        self.current_gen = 0
        self.total_gen = 0
        self.chaotic_val = 0.5

    def sobol_sample(self, n):
        def sobol(i, d):
            v = np.zeros(d)
            i = i + 1
            for j in range(d):
                v[j] = (i >> j) & 1
            return v / (1 << d)

        return [sobol(self.sobol_state + i, len(self.param_space)) for i in range(n)]  
    
    def select_parents(self, scored_population, best_score):
        tournament = random.sample(scored_population, 3)
        parent1 = max(tournament, key=lambda x: x[1])[0]

        def logistic_map(x, r=3.99):
            return r * x * (1 - x)

        self.chaotic_val = logistic_map(getattr(self, "chaotic_val", 0.5))
        chaos_idx = int(self.chaotic_val * len(scored_population)) % len(scored_population)
        parent2 = scored_population[chaos_idx][0]
        score_at_chaos = scored_population[chaos_idx][1]
        if score_at_chaos < 0.8 * best_score:
            parent2 = random.choice(scored_population)[0]
        return parent1, parent2


    def create_individual(self):
        individual = {}
        for param, config in self.param_space.items():
            if config['type'] == 'float':
                individual[param] = round(random.uniform(config['min'], config['max']), 4)
            elif config['type'] == 'int':
                individual[param] = random.randint(config['min'], config['max'])
            elif config['type'] == 'categorical':
                individual[param] = random.choice(config['options'])
        return individual
    
    def breed(self, parent1, parent2):
        try:
            child = {}
            params = list(self.param_space.keys())
            half = len(params) // 2

            for i, param in enumerate(params):
                config = self.param_space[param]
                p1, p2 = parent1[param], parent2[param]

                if config['type'] in ['float', 'int']:
                    if i < half:
                        α = 0.5
                        low = min(p1, p2) - α * abs(p1 - p2)
                        high = max(p1, p2) + α * abs(p1 - p2)
                        val = random.uniform(low, high)
                    
                    else:
                        η = 15
                        u = random.random()
                        if abs(p1 - p2) < 1e-12:
                            val = p1
                        else:
                            if u <= 0.5:
                                β = (2 * u) ** (1 / (η + 1))
                            else:
                                β = (1 / (2 * (1 - u))) ** (1 / (η + 1))
                            val = 0.5 * ((1 + β) * p1 + (1 - β) * p2)

                    val = max(config['min'], min(config['max'], val))
                    if config['type'] == 'int':
                        val = int(round(val))

                    child[param] = val

                elif config['type'] == 'categorical':
                    child[param] = random.choice([p1, p2])

            return self.mutate(child)
        except Exception as e:
            print(f"Breeding failed: {str(e)}")
            # Fallback: return random individual
            return self.create_individual()
        
    def mutate(self, individual):
        for param, config in self.param_space.items():
            if config['type'] == 'categorical':
                options = [o for o in config['options'] if o != individual[param]]
                individual[param] = random.choice(options)
                continue

            if random.random() < 0.05:
                new_val = (
                    random.uniform(config['min'], config['max'])
                    if config['type'] == 'float' else
                    random.randint(config['min'], config['max'])
                )
            else:
                if random.random() < 0.2:
                    σ = 0.3 * (config['max'] - config['min'])
                    new_val = individual[param] + random.gauss(0, σ)
                else:
                    scale = 0.5 * (config['max'] - config['min'])
                    new_val = individual[param] + np.random.standard_cauchy() * scale

            new_val = max(config['min'], min(config['max'], new_val))
            individual[param] = round(new_val, 4) if config['type'] == 'float' else int(round(new_val))

        return individual
    
    def evolve(self, scored_population):
        if len(scored_population) < 2:
            return [self.create_individual() for _ in range(self.population_size)]

        sorted_population = sorted(scored_population, key=lambda x: x[1], reverse=True)
        elites = [ind for ind, score in sorted_population[:self.elite_size]]

        selection_pool = [ind for ind, score in sorted_population[:self.population_size//2]]
        if not selection_pool:
            selection_pool = [self.create_individual() for _ in range(self.population_size)]

        new_generation = elites.copy()

        while len(new_generation) < self.population_size:
            parent1, parent2 = random.sample(selection_pool, 2)
            child = self.breed(parent1, parent2)
            new_generation.append(child)

        return new_generation[:self.population_size]
