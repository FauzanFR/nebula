"""
NebulaTunerEngine

Core optimization driver that coordinates:
- Parallel parameter evaluation
- Genetic optimization cycle
- Adaptive sampling
- Early stopping
"""

import gc
import logging
import os
import random
import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

from .earlystop import EarlyStop
from .optimizer import GeneticOptimizer
from .predictor import NeuralPredictor, predict_njit, train_njit, forward
from .sampler import AdaptiveSampler
from .helper import NebulaInitializationError, save_best_config, setup_logger, strip_result_keys, normalize

def warmup_engine(input_dim: int = 8, batch_size: int = 64):
    X = np.ones((batch_size, input_dim))
    y = np.ones((batch_size, 1))
    W1 = np.ones((input_dim, 16)) * 0.01
    W2 = np.ones((16, 1)) * 0.01
    lr = 0.01

    _ = predict_njit(X, W1, W2)
    _ = forward(X, W1, W2)
    _ = train_njit(X, y, W1, W2, lr, 1)

class NebulaTunerEngine:
    def __init__(self, param_space, genetic_config, param_class, class_df,name,results_path, conf_path="conf.json", batch_size=1000, core_use=None, verbose=True):
        """
        Initializes the optimization engine.
        
        Args:
            param_space: Parameter search space definition
            genetic_config: Configuration for genetic optimization
            param_class: Fixed strategy parameters
            class_df: Strategy class being optimized
            name: Experiment identifier
            results_path: Path for evaluation logs
            conf_path: Path for best configurations
            batch_size: Parallel evaluation batch size
            core_use: CPU cores to utilize (None=all)
            verbose: Control output verbosity
        """
        
        self.ga_optimizer = GeneticOptimizer(param_space, **genetic_config)
        self.sampler = AdaptiveSampler(param_space)
        self.predictor = NeuralPredictor(param_space)
        self.batch_size = batch_size
        self.param_class = param_class
        self.class_df = class_df
        self.name = name
        log_dir = os.path.dirname(results_path)
        self.logger = setup_logger(name, log_dir=log_dir)
        self.conf_path = os.path.join(log_dir, conf_path)
        self.verbose = verbose
        self.initializer(class_df, param_class)
        self.memo_cache = {}
        self.results = []
        self.temp_results = [] 
        self.best_score_ever = 0
        self.core_use = core_use

        self.logger.setLevel(
            logging.DEBUG if verbose else logging.WARNING
        )
        self.logger.debug(f"Use {self.core_use} core CPU")
        warmup_engine()

    def initializer(self, sample_case, params_dict):
        try:
            self.strategy = sample_case(**params_dict)

        except TypeError as e:
            self.logger.error(f"Parameter mismatch in {sample_case.__name__}: {str(e)}")
            valid_params = {k: v for k, v in params_dict.items() 
                        if k in sample_case.__init__.__code__.co_varnames}
            self.strategy = sample_case(**valid_params)

        except Exception as e:
            self.logger.critical(f"Failed to initialize strategy: {str(e)}")
            raise NebulaInitializationError(f"Strategy initialization failed: {str(e)}")

    def run_parallel_batch(self, population):
        def _evaluate_local(params):
            key = tuple(sorted(params.items()))
            if key in self.memo_cache:
                return {**params, 'score': self.memo_cache[key], 'trial_time': 0.0}
            
            try:
                start = time.perf_counter()

                if hasattr(self, 'strategy'):
                    result = self.strategy.run(**params)
                else:
                    strategy = self.class_df(**self.param_class)
                    result = strategy.run(**params)
                    
                duration = time.perf_counter() - start

                if not isinstance(result, (int, float)) or np.isnan(result) or np.isinf(result):
                    raise ValueError("Invalid score result")
                
                self.memo_cache[key] = result
                return {**params, 'score': result, 'trial_time': duration}
            
            except Exception as e:
                self.logger.error(f"Evaluation crash with params {params}: {str(e)}", exc_info=True)
                return None
        try:
            if self.verbose:
                batch_results = list(
                    tqdm(
                        Parallel(n_jobs=self.core_use, backend="loky", batch_size=self.batch_size)(
                            delayed(_evaluate_local)(p) for p in population
                        ),
                        total=len(population)
                    )
                )
            else:
                batch_results = Parallel(n_jobs=self.core_use, backend="loky", batch_size=self.batch_size)(
                    delayed(_evaluate_local)(p) for p in population
                )
        except Exception as e:
            self.logger.critical(f"Parallel processing failed: {str(e)}")
            return [self.ga_optimizer.create_individual() for _ in population]
        
        valid_results = [res for res in batch_results if res is not None]

        if len(valid_results) < len(population) // 2:
            valid_results += [
                {'score': -1e9, **self.ga_optimizer.create_individual()}
                for _ in range(len(population) - len(valid_results))
            ]

        return valid_results
    
    def optimize(self, generations=10, results_path=f"log_results.csv", early_stopping=True):
        try:
            start_time = time.perf_counter()
            terminator = EarlyStop()
            self.ga_optimizer.total_gen = generations
            self.logger.info(f"Starting optimization for {self.name}")
            valid_keys = list(self.ga_optimizer.param_space.keys())
            sobol_points = self.ga_optimizer.sobol_sample(self.ga_optimizer.population_size)
            population = []
            best_save = {'score': -float('inf')}
            save_x = 0
            switch_early_stopping = False

            for point in sobol_points:
                individual = {}
                for i, (param, config) in enumerate(self.ga_optimizer.param_space.items()):
                    if config['type'] == 'float':
                        individual[param] = round(
                            config['min'] + point[i]*(config['max']-config['min']), 4)
                    elif config['type'] == 'int':
                        individual[param] = int(
                            config['min'] + point[i]*(config['max']-config['min']))
                    elif config['type'] == 'categorical':
                        individual[param] = random.choice(config['options'])
                population.append(individual)
            
            self.ga_optimizer.total_gen = generations
            self.ga_optimizer.current_gen = 0

            for gen in range(generations):
                self.ga_optimizer.current_gen = gen
                batch_results = []
                for chunk in self.chunked_population(population):
                    batch_results += self.run_parallel_batch(chunk)

                X_train = np.array([
                    normalize(strip_result_keys(res, self.ga_optimizer.param_space.keys()), self.ga_optimizer.param_space)
                    for res in batch_results
                ])

                y_train = np.array([res['score'] for res in batch_results])
                max_score = np.max(y_train)

                if max_score == 1.0 or max_score == 0.0:
                    x = 1
                else:
                    x = 10 ** int(np.log10(max_score) + 1)
                    save_x = max(save_x, x * 1000)

                y_train = y_train / save_x
                self.predictor.train(X_train, y_train)

                self.temp_results.extend(batch_results)
                if (gen + 1) % 10 == 0 or (gen + 1) == generations:
                    self._save_checkpoint(results_path)
                    gc.collect()
                
                valid_keys = list(self.ga_optimizer.param_space.keys())
                scored_population = [(strip_result_keys(res, valid_keys), res['score']) for res in batch_results]
                self.sampler.update(scored_population,gen+1)
                
                sorted_population = sorted(scored_population, key=lambda x: x[1], reverse=True)
                elites = [ind for ind, _ in sorted_population[:self.ga_optimizer.elite_size]]
                self.ga_optimizer.history += [{"gen": gen, "params": strip_result_keys(res, valid_keys), "score": res["score"]} for res in batch_results]
                
                best = max(batch_results, key=lambda x: x['score'])
                if gen == 0:
                    self.best_score_ever = best['score']
                else:
                    self.best_score_ever = max(self.best_score_ever, best['score'])


                new_generation = elites.copy()
                for _ in range(self.ga_optimizer.population_size - self.ga_optimizer.elite_size):
                    parent1, parent2 = self.ga_optimizer.select_parents(scored_population,self.best_score_ever)
                    child = self.ga_optimizer.breed(parent1, parent2)
                    new_generation.append(child)
                
                X_pred = np.array([normalize(c, self.ga_optimizer.param_space) for c in new_generation])
                pred_scores = self.predictor.predict(X_pred).flatten()

                top_indices = np.argsort(pred_scores)[::-1][:self.ga_optimizer.population_size - len(elites)]
                selected_children = [new_generation[i] for i in top_indices]

                population = elites + selected_children

                if batch_results:
                    best_save = max(batch_results, key=lambda x: x['score'])
                    self.logger.info(f"Gen {gen+1} | Best Score: {best_save['score']:.2f}")
                    
                if terminator.should_stop(best_save['score']) and gen+1 < len(range(generations)):
                    if early_stopping:
                        save_best_config(self.name, best_save, self.conf_path)
                        total_time = time.perf_counter() - start_time
                        self._save_final(results_path, total_time=total_time)
                        switch_early_stopping = True
                        break
                    else:
                        terminator.patience = 3
                        inject_ratio = min(0.5, 0.1 + terminator.total_stagnant * 0.05)
                        inject_size = int(self.ga_optimizer.population_size * inject_ratio)
                        self.logger.info(f"Stagnan, Inject {inject_size} individu...")
                        population[-inject_size:] = [self.sampler.sample() for _ in range(inject_size)]

            if switch_early_stopping == False:
                save_best_config(self.name, best_save, self.conf_path)
                total_time = time.perf_counter() - start_time
                self._save_final(results_path, total_time=total_time)

        except Exception as e:
            self.logger.critical(f"Optimization crashed: {str(e)}")
            self._save_final(results_path, total_time="N/A (crashed)")
            raise
            
    def _save_checkpoint(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True) 
            pd.DataFrame(self.temp_results).to_csv(
                path + ".tmp", 
                mode='a', 
                index=False, 
                header=not os.path.exists(path + ".tmp")
            )
            self.temp_results = [] 
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def _save_final(self, path, total_time=None):
        try:
            if os.path.exists(path + ".tmp"):
                print('a')
                tmp_df = pd.read_csv(path + ".tmp")
                final_df = pd.concat([tmp_df, pd.DataFrame(self.temp_results)])
                final_df.to_csv(path, index=False)
                os.remove(path + ".tmp")
                if total_time is not None:
                    final_df["elapsed_time"] = total_time
            else:
                final_df = pd.DataFrame(self.temp_results)
                if total_time is not None:
                    final_df["elapsed_time"] = total_time
                final_df.to_csv(path, index=False)


            print(f"Results saved to {path}")
        except Exception as e:
            print(f"Error saving final results: {str(e)}")

    def chunked_population(self, population):
        for i in range(0, len(population), self.batch_size):
            yield population[i:i + self.batch_size]
    
    def update_results(self, batch_results, path):
        self.results += batch_results
        if self.results:
            pd.DataFrame(self.results).to_csv(path, mode='a', index=False, header=not pd.io.common.file_exists(path))
