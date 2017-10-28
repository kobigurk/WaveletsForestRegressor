```
usage: runner.py [-h] [--config CONFIG] [--log LOG] [--regressor REGRESSOR]
                 [--trees TREES] [--features FEATURES] [--depth DEPTH]
                 [--seed SEED] [--criterion CRITERION] [--bagging BAGGING]
                 [--data DATA] [--labels LABELS] [--results RESULTS] [--shell]

WaveletsForestRegressor runner. Use "python -m pydoc random_forest" or see
"random_forest.html" for more details.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG
  --log LOG             Logging level. Default is INFO.
  --regressor REGRESSOR
                        Regressor type. Either "rf" or
                        "decision_tree_with_bagging". Default is "rf".
  --trees TREES         Number of trees in the forest. Default is 5.
  --features FEATURES   Features to consider in each split. Same options as
                        sklearn's DecisionTreeRegressor.
  --depth DEPTH         Maximum depth of each tree. Default is 9. Use 0 for
                        unlimited depth.
  --seed SEED           Seed for random operations. Default is 2000.
  --criterion CRITERION
                        Splitting criterion. Same options as sklearn's
                        DecisionTreeRegressor. Default is "mse".
  --bagging BAGGING     Bagging. Only available when using the
                        "decision_tree_with_bagging" regressor. Default is
                        0.8.
  --data DATA           Training data csv path. Default is "trainingData.csv".
  --labels LABELS       Training labels csv path. Default is
                        "trainingLabel.csv".
  --results RESULTS     Results save path.
  --shell               Drop into python shell after calculating smoothness.
                        Default is False.
```
