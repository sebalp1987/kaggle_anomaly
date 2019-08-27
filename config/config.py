gral_parameters = {
    'random_state': 42,
    'fbeta_score': 2,
    'sampling': None
}

stratified_kfolds = {
    'n_splits': 5,
    'shuffle': False,
    'random_state': gral_parameters.get('random_state')
}

oversampling = {
    'ratio': 'minority',
    'n_neighbors': 200,
    'random_state': gral_parameters.get('random_state')
}

decision_tree_optimal = {

    'criterion': 'gini',
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_features': None,
    'class_weight': None,
    'sampling': None,
    'random_state': gral_parameters.get('random_state')

}

random_forest_optimal = {
    'n_estimators': 300,
    'criterion': 'gini',
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_features': None,
    'class_weight': 'balanced_subsample',
    'sampling': None,
    'random_state': gral_parameters.get('random_state')

}

lgb_optimal = {
    'boosting_type': 'goss',
    'max_leaves': 300,
    'max_depth': -1,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'objective': 'binary',
    'class_weight': 'balanced',
    'sampling': None,
    'random_state': gral_parameters.get('random_state')

}