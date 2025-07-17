import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def save_results(mse, mae, r2, rmse,
                 train_r2, model_name, confirm=True):
    """Saves the results of the model to a csv file.

    Args:
        mse (_type_): mean squared error
        mae (_type_): mean absolute error
        r2 (_type_): r-squared
        rmse (_type_): root mean squared error
        train_r2 (_type_): r-squared for the train set
        model_name (_type_): name of the model
    """

    # Check if the file exists
    file_exists = os.path.isfile('results.csv')
    # Create a dataframe with the results
    results = pd.DataFrame({'Model': model_name,
                            'Mean Squared Error (lower the better)': mse,
                            'Mean Absolute Error (lower the better)': mae,
                            'Root Mean Squared Error (lower the better)': rmse,
                            'R² (closer to 1 the better)': r2,
                            'Train R² (closer to 1 the better)': train_r2}, index=[0])

    # Save the results to a csv file
    if file_exists:
        results.to_csv('results.csv', mode='a', header=False, index=False)
    else:
        results.to_csv('results.csv', mode='w', header=True, index=False)
    results = pd.read_csv('results.csv').drop_duplicates()
    results.to_csv('results.csv', mode='w', header=True, index=False)
    # Confirm the results were saved
    if confirm:
        print('Results saved successfully to "results.csv".')


def plot_results(regression_model: str = 'All Models',
                 plot_height: int = 8,
                 save_plot: bool = False):
    """Plots the metrics saved to results.csv

    Args:
        regression_model (str, optional): The models you want on the plot. Defaults to 'All Models'.
        plot_height (int, optional): Height of the plot (useful when there are many models to plot). Defaults to 8.
        save_plot (bool, optional): Saves the plot as a png file. Defaults to False.
    """
    results = pd.read_csv('results.csv')
    if regression_model == 'All Models':
        pass
    else:
        results = results.loc[results['Model'].str.contains(regression_model)]

    results.plot(kind='barh', x='Model',
                 title='Model Performance',
                 figsize=(12, plot_height), grid=True)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylabel('Model')
    plt.xlabel('Performance Metric')
    plt.xticks(np.arange(0, 1.05, step=0.05))
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')

    if save_plot:
        plt.savefig(f'results.png', bbox_inches='tight', dpi=300)
        print(f'Plot saved as results.png')

    plt.show()
