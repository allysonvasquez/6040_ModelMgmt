# calculate the entropy for a dataset
from math import log2

# calculate the entropy for the split in the dataset
def entropy(y_class, n_class):
    return - (y_class * log2(y_class) + n_class * log2(n_class))

entropy(class_bal['yes'], class_bal['no'])