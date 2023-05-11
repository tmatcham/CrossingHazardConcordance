# CrossingHazardConcordance

Code is provided to replicate the experiments in chapters 4, 5, and 6.

Chapter 6 uses DeepHit code from https://github.com/chl8856/DeepHit with changes made to also include our concordance index in the loss and validation. To select which concordance index is used in the loss function set values of beta and delta in main_RandomSearch.py. To set which concordance is used in validation comment out all but the chosen va_result1 in get_main.py.
