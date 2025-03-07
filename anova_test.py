from scipy.stats import f_oneway

def perform_anova(accuracies):
    """
    Perform ANOVA test on the accuracies of multiple models.

    Args:
        accuracies (dict): Dictionary where keys are model names and values are lists of accuracies.

    Returns:
        f_stat (float): F-statistic from the ANOVA test.
        p_value (float): p-value from the ANOVA test.
        anova_result (str): Interpretation of the ANOVA result.
    """
    # Extract accuracy lists from the dictionary
    anova_data = [accuracies[model] for model in accuracies]

    # Perform ANOVA
    f_stat, p_value = f_oneway(*anova_data)

    # Interpret the result
    if p_value < 0.05:
        anova_result = "Significant difference found between models (p < 0.05)."
    else:
        anova_result = "No significant difference found between models (p >= 0.05)."

    return f_stat, p_value, anova_result