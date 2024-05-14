import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from scipy.stats import loguniform, uniform
import time

# Read data
df = pd.read_csv('students.csv')
directory = 'students_plots'

# Prepare data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - mean) / std

# Function to perform search
def perform_search(name, model, params, search_type, X_train, y_train, X_test, y_test):
    search = None  # Initialize search variable
    start_time = time.time()

    if params:
        if search_type == 'grid':
            search = GridSearchCV(model, params, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, params, n_iter=100, cv=10, scoring='neg_root_mean_squared_error',
                                        random_state=42, return_train_score=True)
        elif search_type == 'bayes':
            search = BayesSearchCV(model, params, n_iter=100, cv=10, scoring='neg_root_mean_squared_error',
                                   random_state=42, return_train_score=True)

    if search:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f'Best parameters for {name} ({search_type} search): {search.best_params_}')

        search_df = pd.DataFrame(search.cv_results_)
        search_df['Model'] = name
        search_df['Search Type'] = search_type
    else:
        best_model = model.fit(X_train, y_train)

    end_time = time.time()
    duration = end_time - start_time
    print(f'Time taken for {name} ({search_type} search): {end_time - start_time:.8f} seconds')
    return best_model, search_df if search else best_model, duration



# List of models and their parameters
models = [
    ('Linear Regression', LinearRegression(), {}, 'grid'),
    ('Ridge', Ridge(), {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, 'grid'),
    ('Lasso', Lasso(), {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, 'grid'),
    ('ElasticNet', ElasticNet(),
     {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
     'grid'),
    ('Linear Regression', LinearRegression(), {}, 'random'),
    ('Ridge', Ridge(), {'alpha': loguniform(1e-4, 1e2)}, 'random'),
    ('Lasso', Lasso(), {'alpha': loguniform(1e-4, 1e2)}, 'random'),
    ('ElasticNet', ElasticNet(), {'alpha': loguniform(1e-4, 1e2), 'l1_ratio': uniform(0.1, 0.9)}, 'random'),
    ('Linear Regression', LinearRegression(), {}, 'bayes'),
    ('Ridge', Ridge(), {'alpha': (1e-4, 1e2)}, 'bayes'),
    ('Lasso', Lasso(), {'alpha': (1e-4, 1e2)}, 'bayes'),
    ('ElasticNet', ElasticNet(), {'alpha': (1e-4, 1e2), 'l1_ratio': (0.1, 0.9)}, 'bayes')
]

# List to store results
results_combined = []

# Iterate over models
for name, model, params, search_type in models:
    best_model, search_results, duration = perform_search(name, model, params, search_type, X_train, y_train, X_test, y_test)

    # Evaluate model performance
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    cv = cross_val_score(estimator=best_model, X=X_train, y=y_train, cv=10, scoring='neg_root_mean_squared_error')
    test_score = best_model.score(X_test, y_test)
    train_score = best_model.score(X_train, y_train)
    # print type of search results


    # Store results
    result = {
        'Model': name,
        'Test Score': test_score,
        'Train Score': train_score,
        'RMSE': rmse,
        'Cross Validation Score': -cv.mean(),
        'Search Type': search_type,
        'time taken': duration,
    }
    results_combined.append(result)


# Convert results to DataFrame
results_combined_df = pd.DataFrame(results_combined)

# Save results to CSV
results_combined_df.to_csv('results_combined.csv', index=False)

# Print overall results
print('Overall Results:')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(results_combined_df)
