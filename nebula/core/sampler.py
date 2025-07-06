"""
AdaptiveSampler

Generates new parameters by:
- Learning from high-performing historical parameters
- Adapting categorical distributions dynamically
- Balancing exploration and exploitation
"""

import random

import numpy as np

class AdaptiveSampler:
    def __init__(self, param_space):
        self.param_space = param_space
        self.history = [] 
        self.categorical_counts = {}  
        
        for param, config in param_space.items():
            if config['type'] == 'categorical':
                self.categorical_counts[param] = {opt: 0 for opt in config['options']}
    
    def update(self, scored_population, gen):
        if not scored_population:
            return
        
        scored_population_sorted = sorted(scored_population, key=lambda x: x[1], reverse=True)
        top_size = max(1, len(scored_population_sorted) // (5 if gen < 10 else 10))
        top_params = [params for params, _ in scored_population_sorted[:top_size]]

        for params in top_params:
            if all(k in params for k in self.param_space):
                self.history.append(params)
        
        for params in top_params:  
            for param, counter in self.categorical_counts.items():
                if param in params:
                    counter[params[param]] += 1
    
    def sample(self):
        """
        Generate a new parameter set combining:
        - Historical best performers (50%)
        - Local exploration (30%)
        - Global exploration (20%)
        
        Returns:
            dict: New parameter combination
        """
        individual = {}
        use_history = len(self.history) > 10 and random.random() > 0.5 
        for param, config in self.param_space.items():
            if config['type'] == 'float':
                if self.history and random.random() > 0.3:
                    vals = [p[param] for p in self.history if param in p]
                    if vals:
                        mean, std = np.mean(vals), np.std(vals)
                        val = np.clip(np.random.normal(mean, std), config['min'], config['max'])
                        individual[param] = round(val, 4)
                        continue

                val = np.random.beta(2, 2) * (config['max'] - config['min']) + config['min']
                individual[param] = round(val, 4)
                
            elif config['type'] == 'int':
                if use_history:
                    vals = [p[param] for p in self.history if param in p]
                    if vals:
                        individual[param] = max(set(vals), key=vals.count)
                        continue
                individual[param] = random.randint(config['min'], config['max'])
                
            elif config['type'] == 'categorical':
                if param in self.categorical_counts and sum(self.categorical_counts[param].values()) > 0:
                    options, counts = zip(*self.categorical_counts[param].items())
                    individual[param] = random.choices(options, weights=counts)[0]
                else:
                    individual[param] = random.choice(config['options'])
        
        return individual