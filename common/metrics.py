from numpy import linalg as LA


def behavior_dist(behavior, loader):
    """
    Calculates minimum l2 distance between real behavior vector and generated behavior vector.

    Parameters
    ----------
    behavior: generated vector
    loader: torch loader with real behavior samples
    """
    min_dist = -1
    for X_batch, Y_batch in loader:
        for sample in X_batch:
            dist = LA.norm(sample.numpy() - behavior)
            min_dist = dist if dist > min_dist else min_dist
    return min_dist
