import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt




def plot_and_save(X_test, predictions, feature_names, colors, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # For each feature in the dataset
    for i in range(X_test.shape[1]):
        # Extract the feature values from the test data
        feature_values_test = X_test[:, i]

        # Calculate the coefficients of the line of best fit
        coefficients = np.polyfit(feature_values_test, predictions, 1)

        # Create a function that represents the line of best fit
        line_of_best_fit = np.poly1d(coefficients)

        # Plot the feature values against the Chance of Admit
        plt.scatter(feature_values_test, predictions, color=colors[i % len(colors)])
        plt.plot(feature_values_test, line_of_best_fit(feature_values_test), color='black', label=f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

        plt.xlabel(feature_names[i])
        plt.ylabel('Chance of Admit')
        plt.title(f'{feature_names[i]} vs Chance of Admit')
        plt.legend()
        plt.savefig(f'{directory}/{feature_names[i]}_vs_Chance_of_Admit.png')

        # plt.show()

def extract_features_target(df):

    return X, y