import numpy as np
from nebula.nebula_tune import space, Tuner, genetic_config

# Define your parameter search space
param_space = [
    # Basic Strategy (4)
    space.float("aggressiveness", 0.0, 1.0),
    space.float("defensiveness", 0.0, 1.0),
    space.float("resourcefulness", 0.1, 2.0),
    space.float("reaction_time", 0.01, 0.5),

    # Sensors & Vision (4)
    space.int("vision_range", 5, 50),
    space.float("radar_sensitivity", 0.1, 1.0),
    space.cat("thermal_scope", ["off", "basic", "advanced"]),
    space.cat("motion_detection", ["none", "passive", "active"]),

    # Tactical AI (5)
    space.cat("pathfinding", ["A*", "Dijkstra", "RRT", "random"]),
    space.float("learning_rate", 0.001, 0.1),
    space.float("exploration_prob", 0.0, 1.0),
    space.int("memory_depth", 1, 20),
    space.cat("prediction_model", ["none", "linear", "mlp"]),

    # Movement & Mobility (4)
    space.float("max_speed", 1.0, 10.0),
    space.float("acceleration", 0.1, 5.0),
    space.float("jump_power", 0.0, 3.0),
    space.int("dash_frequency", 1, 10),

    # Energy & Battery (3)
    space.int("battery_capacity", 1000, 10000),
    space.float("energy_efficiency", 0.1, 1.0),
    space.cat("auto_charge", ["off", "slow", "fast"]),

    # Weaponry (6)
    space.cat("weapon_type", ["laser", "bullet", "plasma", "railgun"]),
    space.float("fire_rate", 1.0, 10.0),
    space.float("reload_speed", 0.5, 5.0),
    space.int("ammo_capacity", 10, 200),
    space.cat("targeting_algo", ["simple", "lead", "adaptive"]),
    space.float("lock_on_time", 0.1, 2.0),

    # Communication & Team (4)
    space.float("signal_strength", 0.1, 1.0),
    space.cat("team_strategy", ["solo", "support", "leader"]),
    space.int("ping_interval", 1, 30),
    space.cat("message_compression", ["none", "lz", "huffman"]),
]

# Create a class wrapper for your evaluation function
# Nebula expects the evaluation function to be a method of a class
class MyEvaluator:
    def __init__(self):
        # You might initialize datasets or other common resources here
        pass
    
    def run(
        self,
        aggressiveness, resourcefulness, defensiveness, reaction_time,
        vision_range, radar_sensitivity, thermal_scope, motion_detection,
        pathfinding, learning_rate, exploration_prob, memory_depth, prediction_model,
        max_speed, acceleration, jump_power, dash_frequency,
        battery_capacity, energy_efficiency, auto_charge,
        weapon_type, fire_rate, reload_speed, ammo_capacity, targeting_algo, lock_on_time,
        signal_strength, team_strategy, ping_interval, message_compression
    ):
        score = (
            4 * (aggressiveness * resourcefulness) +
            3 * (defensiveness * (1 - reaction_time)) +
            1.5 * vision_range +
            0.8 * radar_sensitivity +
            2 * (energy_efficiency * battery_capacity / 1000) +
            1.2 * max_speed +
            2.5 * (fire_rate * (200 - reload_speed)) +
            0.5 * jump_power +
            0.6 * acceleration +
            0.7 * dash_frequency +
            0.3 * memory_depth +
            0.2 * exploration_prob +
            0.4 * signal_strength +
            0.05 * ping_interval +
            2.0 * (1 if auto_charge == "fast" else 0) +
            1.5 * (1 if thermal_scope == "advanced" else 0) +
            1.0 * (1 if motion_detection == "active" else 0) +
            2.0 * (1 if prediction_model == "mlp" else 0) +
            2.0 * (1 if team_strategy == "leader" else 0) +
            2.0 * (1 if targeting_algo == "adaptive" else 0) +
            np.random.normal(0, 3.0)
        )
        return round(score, 4)



# Define genetic algorithm configuration (optional, defaults are provided)
ga_config = genetic_config(population_size=100, elite_size=3, mutation_rate=0.35)

# Run the tuner
if __name__ == "__main__":
    Tuner(
        param_space=param_space,
        genetic_config=ga_config,
        param_class=None,
        name="game_balancing",
        generations=10,
        batch_size=100,
        class_df=MyEvaluator, 
        verbose=True,
        results_path="log/game.csv"
        )