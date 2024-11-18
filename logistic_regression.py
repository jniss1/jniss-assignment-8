import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt  # Import pyplot with the Agg backend

# Directory to save results
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def create_clustered_data(shift_value, num_samples=100, cluster_variability=0.5):
    """
    Generate two elliptical clusters with a specified shift distance between them.
    """
    np.random.seed(0)
    
    # Covariance matrix for generating ellipsoidal clusters
    cov_matrix = np.array([[cluster_variability, cluster_variability * 0.8], 
                           [cluster_variability * 0.8, cluster_variability]])
    
    # Generate class 0 data (no shift)
    class_0_data = np.random.multivariate_normal(mean=[1, 1], cov=cov_matrix, size=num_samples)
    class_0_labels = np.zeros(num_samples)

    # Generate class 1 data with a shift by `shift_value`
    class_1_data = np.random.multivariate_normal(mean=[1 + shift_value, 1 + shift_value], cov=cov_matrix, size=num_samples)
    class_1_labels = np.ones(num_samples)

    # Combine both classes into a single dataset
    X_data = np.vstack((class_0_data, class_1_data))
    y_labels = np.hstack((class_0_labels, class_1_labels))
    
    return X_data, y_labels

def fit_logistic_model(X, y):
    """
    Fit a logistic regression model and return the coefficients (Beta0, Beta1, Beta2).
    """
    model = LogisticRegression()
    model.fit(X, y)
    intercept = model.intercept_[0]
    coef_1, coef_2 = model.coef_[0]
    
    return model, intercept, coef_1, coef_2

def do_experiments(start_val, end_val, num_steps):
    """
    Run experiments across a range of shift distances and log the results.
    """
    # Generate an array of shift distances for the experiment
    shift_distances = np.linspace(start_val, end_val, num_steps)
    
    # Lists to hold results for plotting
    intercept_vals, coef1_vals, coef2_vals = [], [], []
    slopes, intercept_ratios, logistic_losses, margin_widths = [], [], [], []

    # To store experiment data (for later use in the plots)
    experiment_results = {}

    # Plot layout setup for dataset visualization
    num_cols = 2
    num_rows = (num_steps + num_cols - 1) // num_cols
    plt.figure(figsize=(20, num_rows * 10))

    # Run experiments for each shift distance and gather the results
    for idx, shift in enumerate(shift_distances, 1):
        X, y = create_clustered_data(shift_value=shift)
        model, intercept, coef_1, coef_2 = fit_logistic_model(X, y)

        # Store model coefficients for later analysis
        intercept_vals.append(intercept)
        coef1_vals.append(coef_1)
        coef2_vals.append(coef_2)

        # Compute slope (beta1 / beta2) and intercept ratio (beta0 / beta2)
        if abs(coef_2) > 1e-6:  # Avoid division by zero
            slope = -coef_1 / coef_2
            intercept_ratio = -intercept / coef_2
        else:
            slope = np.nan  # Handle cases where beta2 is 0
            intercept_ratio = np.nan
        
        slopes.append(slope)
        intercept_ratios.append(intercept_ratio)

        # Calculate the logistic loss
        prob_preds = model.predict_proba(X)[:, 1]
        log_loss = -np.mean(y * np.log(prob_preds) + (1 - y) * np.log(1 - prob_preds))
        logistic_losses.append(log_loss)

        # Visualize the data and decision boundary
        plt.subplot(num_rows, num_cols, idx)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='green', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='purple', label='Class 1')
        
        # Generate grid for decision boundary and confidence contours
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        prob_contours = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        prob_contours = prob_contours.reshape(xx.shape)

        # Plot the decision boundary
        plt.contour(xx, yy, prob_contours, levels=[0.5], colors='black')
        
        # Plot confidence intervals (70%, 80%, 90%) for each class
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, prob_contours, levels=[level, 1.0], colors=['purple'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, prob_contours, levels=[0.0, 1 - level], colors=['green'], alpha=alpha)
            if level == 0.7:
                # Compute the margin width (distance between the contours)
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,
                                  class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                margin_distance = np.min(distances)
                margin_widths.append(margin_distance)

        # Annotate the plot with the logistic regression equation and margin width
        plt.title(f"Shift Distance = {shift:.2f}", fontsize=24)
        plt.xlabel("x1")
        plt.ylabel("x2")

        equation_text = f"{intercept:.2f} + {coef_1:.2f} * x1 + {coef_2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept_ratio:.2f}"
        margin_text = f"Margin Width: {margin_distance:.2f}"
        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=24, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=24, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if idx == 1:
            plt.legend(loc='lower right', fontsize=20)

        # Store results for later analysis
        experiment_results[shift] = (X, y, model, intercept, coef_1, coef_2, margin_distance)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plot parameters vs. shift distances
    plt.figure(figsize=(18, 15))

    # Plot Beta0 (Intercept) vs. Shift Distance
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, intercept_vals, marker='o', color='black')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Plot Beta1 vs. Shift Distance
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, coef1_vals, marker='o', color='black')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Plot Beta2 vs. Shift Distance
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, coef2_vals, marker='o', color='black')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Plot Beta1 / Beta2 (Slope) vs. Shift Distance
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slopes, marker='o', color='black')
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1 / Beta2")
    plt.ylim(-2, 0)

    # Plot Beta0 / Beta2 (Intercept Ratio) vs. Shift Distance
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_ratios, marker='o', color='black')
    plt.title("Shift Distance vs Beta0 / Beta2 (Intercept Ratio)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0 / Beta2")

    # Plot Logistic Loss vs. Shift Distance
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, logistic_losses, marker='o', color='black')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Plot Margin Width vs. Shift Distance
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o', color='black')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start_value = 0.25
    end_value = 2.0
    num_steps = 8
    do_experiments(start_value, end_value, num_steps)
