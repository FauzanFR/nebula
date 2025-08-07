from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from nebula.nebula_tune import space, Tuner, genetic_config

param_space = [
    space.int("n_estimators", 10, 300),
    space.int("max_depth", 1, 50),
    space.int("min_samples_split", 2, 20),
    space.int("min_samples_leaf", 1, 20),
    space.cat("max_features", ["sqrt", "log2", None]),
    space.bool("bootstrap"),
    space.cat("criterion", ["gini", "entropy", "log_loss"]),
    space.cat("class_weight", [None, "balanced"]),
    space.int("max_leaf_nodes", 10, 100),
    space.float("min_impurity_decrease", 0.0, 0.1)
]

class RFCase:
    def __init__(self, dataset):
        X, y = dataset(return_X_y=True)
        self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        self.scorer = make_scorer(f1_score, average='macro')

    def run(self, **kwargs):
        clf = RandomForestClassifier(**kwargs, random_state=42, n_jobs=1)
        scores = cross_val_score(clf, self.X_train, self.y_train, cv=3, scoring=self.scorer)
        return round(scores.mean(), 6)

# Nebula expects each parameter name to match the corresponding function argument name 
param_class = {
    'dataset' : load_breast_cancer
}

if __name__ == "__main__":
    Tuner(
        param_space,
        genetic_config(30,4,0.35),
        param_class,
        name="RF",
        generations=10,
        batch_size=30,
        class_df=RFCase,
        verbose=True,
        results_path="log\RF.csv",
        core_use=None,
        early_stopping=True
        )