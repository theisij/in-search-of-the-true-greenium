# Descriptive statistics
def desc_table(data, cols, cols_names, save_as, include_skewness=False,
               canonicalize_zero=False):
    # include 5% and 95% in the percentiles list
    desc = (
        data[cols]
        .describe(percentiles=[.05, .25, .5, .75, .95])
        .transpose()
    )
    # drop the min/max and keep the 5%/95% instead
    desc = desc.drop(columns=['min','max'])

    # Add skewness if requested
    if include_skewness:
        skewness = data[cols].skew()
        desc['Skewness'] = skewness

    # Create row names
    desc.index = cols_names

    # Create column names
    if include_skewness:
        desc.columns = ['$N$', 'Mean', 'Std. Dev.', '5\\%', '25\\% ', '50\\%', '75\\%', '95\\%', 'Skewness']
    else:
        desc.columns = ['$N$', 'Mean', 'Std. Dev.', '5\\%', '25\\% ', '50\\%', '75\\%', '95\\%']

    # Optionally canonicalize near-zero numeric residuals so they render as
    # "0.00" instead of "-0.00" under %.2f. Useful for standardized series
    # whose mean is mathematically zero but lands on -1e-16 vs +1e-16
    # depending on FP summation order. Default off so other callers
    # (esg_descriptives, char_descriptives, etc.) keep current behavior.
    if canonicalize_zero:
        num_cols = [c for c in desc.columns if c != '$N$']
        # round(-1e-16, 10) gives -0.0; adding 0.0 turns -0.0 into 0.0 per
        # IEEE 754. Genuine non-zero negatives (e.g. -3.55) are unaffected.
        desc[num_cols] = desc[num_cols].round(10) + 0.0

    # N to integer
    desc['$N$'] = desc['$N$'].astype(int).apply(lambda x: f"{x:,}")

    # To latex
    desc.to_latex(save_as, float_format="%.2f")
    return None