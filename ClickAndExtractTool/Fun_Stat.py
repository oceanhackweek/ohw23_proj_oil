#
# Statistics
#

from statistics import *

#x = data
def get_stats(x):
    return ( 
        mean(x),              # Arithmetic mean (“average”) of data.
        #fmean(x),             # Fast, floating point arithmetic mean, with optional weighting.
        geometric_mean(x),    # Geometric mean of data.
        harmonic_mean(x),     # Harmonic mean of data.
        median(x),            # Median (middle value) of data.
        #median_low(x),        # Low median of data.
        #median_high(x),       # High median of data.
        #median_grouped(x),    # Median, or 50th percentile, of grouped data.
        mode(x),              # Single mode (most common value) of discrete or nominal data.
        #multimode(x),         # List of modes (most common values) of discrete or nominal data.
        ",".join(str(x) for x in quantiles(x))          # Divide data into intervals with equal probability.
    )

