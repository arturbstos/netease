"""Slow steps of the analysis, one module per notebook. Importing this package registers every step in thesis.cache.
    prediction   notebook 02 (tuning, models, calibration, robustness, inactive label, history curve)
    recommender  notebook 03 (exposure model, click model and re-ordering, observed re-ordering, first pages)
    checks       appendix A1 (learners, training curves, user weights, creator and variety features)"""
from thesis.heavy import prediction, recommender, checks  # noqa: F401
