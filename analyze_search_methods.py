import pandas as pd
import matplotlib.pyplot as plt
import os


if not os.path.exists('search_methods_plots'):
    os.makedirs('search_methods_plots')

if not os.path.exists('results_combined.csv'):
    # make error message
    print('No results to analyze')
results_combined = pd.read_csv('results_combined.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(results_combined)

search_type_colors = {
    'grid': 'red',
    'random': 'green',
    'bayes': 'blue'
}

model_names = results_combined['Model'].unique()

for model_name in model_names:
    filtered_df = results_combined[results_combined['Model'] == model_name]
    test_score_range = filtered_df['Test Score'].max() - filtered_df['Test Score'].min()

    if test_score_range == 0:
        delta = 0.01
    else:
        delta = test_score_range * 0.1

    plt.figure(figsize=(10, 6))
    for search_type in filtered_df['Search Type'].unique():
        search_type_df = filtered_df[filtered_df['Search Type'] == search_type]
        plt.bar(search_type_df['Search Type'], search_type_df['Test Score'], color=search_type_colors[search_type])
    plt.xlabel('Search Type')
    plt.ylabel('Test Score')
    plt.ylim(max(0, filtered_df['Test Score'].min() - delta),
             filtered_df['Test Score'].max() + delta)
    plt.title(f'Test Score for {model_name} Against Every Search Type')
    plt.savefig(f'search_methods_plots/{model_name.replace(" ", "_")}_test_score.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    for search_type in filtered_df['Search Type'].unique():
        search_type_df = filtered_df[filtered_df['Search Type'] == search_type]
        bars = plt.bar(search_type_df['Search Type'], search_type_df['time taken'],
                       color=search_type_colors[search_type])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')
    plt.xlabel('Search Type')
    plt.ylabel('Time Taken')
    if delta > filtered_df['time taken'].min():
        plt.ylim(0, filtered_df['time taken'].max() + delta)
    else:
        plt.ylim(max(0, filtered_df['time taken'].min() - delta),
                 filtered_df['time taken'].max() + delta)
    plt.title(f'Time Taken for {model_name} Against Every Search Type')
    plt.savefig(f'search_methods_plots/{model_name.replace(" ", "_")}_time_taken.png')
    plt.show()
