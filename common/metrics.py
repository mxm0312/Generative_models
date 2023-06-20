from numpy import linalg as LA
import torch

def behavior_dist(behaviors, loader):
    """
    Calculates minimum l2 distance between real behavior vector and generated behavior vector.

    Parameters
    ----------
    behavior: generated batch of behaviors
    loader: torch loader with real behavior samples
    """
    min_dist = -1.0
    behaviors = behaviors.detach().numpy()
    for beh in behaviors:
        for X_batch, _ in loader:
            for sample in X_batch:
                dist = LA.norm(sample.numpy() - beh)
                min_dist = dist if dist > min_dist else min_dist
    return min_dist

def harmful_behavior_probs(behavior, clf_path):
    clf = torch.load(clf_path)
    clf.eval()
    with torch.no_grad():
        probs = clf(behavior)
    return probs
