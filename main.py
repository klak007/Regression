
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Define a list of colors for each feature
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

df = pd.read_csv('students.csv')
directory = 'students_plots'

X = df.iloc[:, :-1].values  # all columns except the last one
y = df.iloc[:, -1].values  # only the last column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model using the training data
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions using the test data
# predictions = model.predict(X_test)




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
            [LinearRegression(), Ridge(alpha=0.01), Lasso(alpha=0.01), ElasticNet(alpha=0.01, l1_ratio=0.5)]):
        model.fit(X_train[:, i].reshape(-1, 1), y_train)
        predictions = model.predict(X_test[:, i].reshape(-1, 1))

        plt.plot(X_test[:, i], predictions, color=colors[j % len(colors)],
                 label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}, {model}')

    plt.xlabel(feature_names[i])
    plt.ylabel(f'{df.columns[-1]}')
    plt.title(f'{feature_names[i]} vs {df.columns[-1]}')
    plt.legend()
    plt.savefig(f'{directory}/{feature_names[i]}_vs_{df.columns[-1].replace(" ", "_")}.png')

    plt.clf()

# rmse, r2 test, r2 train, cross validation
# make a loop for all models and print the results

for model in [LinearRegression(), Ridge(alpha=0.01), Lasso(alpha=0.01), ElasticNet(alpha=0.01, l1_ratio=0.5)]:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, predictions)))
    cv = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
    print('Model:', model)
    print(f'Test  data      : {model.score(X_test, y_test)}')
    print(f'Train data      : {model.score(X_train, y_train)}')
    print(f'RMSE            : {rmse}')
    print(f'Cross validation: {cv.mean()}')
    print('-----------------------------------')

    # plot actual vs predicted and line y=x
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.plot([0, 1], [0, 1], color='red', lw=1, ls='--')
    plt.title(f'Actual vs Predicted for {model}')
    plt.savefig(f'{directory}/actual_vs_predicted_{model}.png')
    plt.show()
    plt.clf()