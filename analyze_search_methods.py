import pandas as pd
import matplotlib.pyplot as plt

# read results_combined from csv file
results_combined = pd.read_csv('results_combined.csv')

# display the first 5 rows of the dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(results_combined)

search_type_colors = {
    'grid': 'red',
    'random': 'green',
    'bayes': 'blue'
}

# Filter the dataframe for the specific model
model_names = results_combined['Model'].unique()
# Iterate over model names and create a plot for each
for model_name in model_names:
    # Filter the dataframe for the specific model
    filtered_df = results_combined[results_combined['Model'] == model_name]

    # Calculate dynamic y-axis limits
    test_score_range = filtered_df['Test Score'].max() - filtered_df['Test Score'].min()
    if test_score_range == 0:  # All test scores are the same
        delta = 0.01  # Manually set a small delta
    else:
        delta = test_score_range * 0.1  # Adjust this value to change the percentage of the range used for the delta

    # Create a bar plot for test score
    plt.figure(figsize=(10, 6))
    for search_type in filtered_df['Search Type'].unique():
        search_type_df = filtered_df[filtered_df['Search Type'] == search_type]
        plt.bar(search_type_df['Search Type'], search_type_df['Test Score'], color=search_type_colors[search_type])
    plt.xlabel('Search Type')
    plt.ylabel('Test Score')
    plt.ylim(max(0, filtered_df['Test Score'].min() - delta),
             filtered_df['Test Score'].max() + delta)  # Set y-axis limits
    plt.title(f'Test Score for {model_name} Against Every Search Type')
    plt.savefig(f'search_methods_plots/{model_name}_test_score.png')  # Save the figure
    plt.show()

    # Create a bar plot for time taken
    plt.figure(figsize=(10, 6))
    for search_type in filtered_df['Search Type'].unique():
        search_type_df = filtered_df[filtered_df['Search Type'] == search_type]
        bars = plt.bar(search_type_df['Search Type'], search_type_df['time taken'],
                       color=search_type_colors[search_type])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')  # va: vertical alignment
    plt.xlabel('Search Type')
    plt.ylabel('Time Taken')
    if delta > filtered_df['time taken'].min():
        plt.ylim(0, filtered_df['time taken'].max() + delta)  # Set y-axis limits
    else:
        plt.ylim(max(0, filtered_df['time taken'].min() - delta),
                 filtered_df['time taken'].max() + delta)  # Set y-axis limits
    plt.title(f'Time Taken for {model_name} Against Every Search Type')
    plt.savefig(f'search_methods_plots/{model_name}_time_taken.png')  # Save the figure
    plt.show()