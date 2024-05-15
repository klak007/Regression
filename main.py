from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
from collections import Counter

warnings.filterwarnings('ignore', category=FutureWarning)

colors = ['blue'] # , 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'
colors2 = ['olive', 'purple', 'teal', 'pink', 'brown', 'gray', 'orange']

filename = 'boston' # students, diabetes, boston
df = pd.read_csv(f'{filename}.csv')
directory = f'{filename}_plots'


if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(f'{directory}/features'):
    os.makedirs(f'{directory}/features')
if not os.path.exists(f'{directory}/research'):
    os.makedirs(f'{directory}/research')

corr = df.iloc[:, :-1].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corr, mask=mask, square=True, annot=True, fmt=".2f", linewidths=.8, cmap="coolwarm",
                     annot_kws={"size": 30})  # Add annotation with font size 8

    plt.title('Correlation Matrix of Features')
    plt.subplots_adjust(left=0.2, right=1, bottom=0.1, top=0.95)  # Adjust the spacin

    plt.savefig(f'{directory}/research/correlation_matrix.png')
    plt.show()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)


# standardyzacja/ normalizacja danych - odchylenie standardowe = 1, srednia = 0
# czemu nie dzielimy testu przez jego srednia i odchylenie standardowe?
# bo w rzeczywistosci nie znamy sredniej i odchylenia standardowego dla danych testowych
# zawsze korzystamy z tych z danych treningowych

# normalize train data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - mean) / std

feature_names = df.columns[:-1]

if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(X_test.shape[1]):
    plt.scatter(X_test[:, i], y_test, color=colors[i % len(colors)])

    for j, model in enumerate(
            [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.001), ElasticNet(alpha=0.01, l1_ratio=0.1)]):
        model.fit(X_train[:, i].reshape(-1, 1), y_train)
        predictions = model.predict(X_test[:, i].reshape(-1, 1))

        plt.plot(X_test[:, i], predictions, color=colors2[j % len(colors2)],
                 label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}, {model}')

    plt.xlabel(feature_names[i])
    plt.ylabel(f'{df.columns[-1]}')
    plt.title(f'{feature_names[i]} vs {df.columns[-1]}')
    plt.legend()
    plt.savefig(f'{directory}/features/{feature_names[i].replace(" ", "_")}_vs_{df.columns[-1].replace(" ", "_")}.png')
    # plt.show()
    plt.clf()

# Define the parameter grids for the models
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
elastic_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                  'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Ridge', Ridge(), ridge_params),
    ('Lasso', Lasso(), lasso_params),
    ('ElasticNet', ElasticNet(), elastic_params)
]

results = []

for name, model, params in models:
    if params:
        # Perform grid search
        grid_search = GridSearchCV(model, params, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f'Best parameters for {name}: {grid_search.best_params_}')


    else:
        # Fit the model without grid search
        best_model = model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, predictions)))
    cv = cross_val_score(estimator=best_model, X=X_train, y=y_train, cv=10, scoring='neg_root_mean_squared_error')
    test_score = best_model.score(X_test, y_test)
    train_score = best_model.score(X_train, y_train)

    # Store the results in a dictionary
    result = {
        'Model': name,
        'Test Score': test_score,
        'Train Score': train_score,
        'RMSE': rmse,
        'Cross Validation Score': -cv.mean()
    }
    results.append(result)

    # plot actual vs predicted and line y=x
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.plot([0, max(y_test)], [0, max(y_test)], color='red', lw=1, ls='--')
    plt.title(f'Actual vs Predicted for {name}')
    plt.savefig(f'{directory}/research/actual_vs_predicted_{name}.png')
    #plt.show()
    plt.clf()


results_df = pd.DataFrame(results)
print('Overall Results for grid search:')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
print(results_df)

bar_width = 0.4
colours = {'Linear Regression': 'green', 'Ridge': 'blue', 'Lasso': 'orange', 'ElasticNet': 'magenta'}


def plot_scores(df, score_col, title, score_type):
    plt.figure(figsize=(7, 4))
    for model, color in colours.items():
        plt.bar(df[df['Model'] == model]['Model'], df[df['Model'] == model][score_col], color=color, width=bar_width)
    plt.xlabel('Model')
    plt.ylabel(score_col)
    plt.title(title)
    plt.ylim(min(df[score_col]) - 0.01, max(df[score_col]) + 0.01)  # Set y-axis limits
    plt.savefig(f'{directory}/research/{score_type}.png')
    # plt.show()
    plt.clf()


plot_scores(results_df, 'Test Score', 'Test Scores for different models', 'test_scores')
plot_scores(results_df, 'Train Score', 'Train Scores for different models', 'train_scores')
plot_scores(results_df, 'RMSE', 'RMSE for different models', 'rmse')
plot_scores(results_df, 'Cross Validation Score', 'Cross Validation Scores for different models',
            'cross_validation_scores')
