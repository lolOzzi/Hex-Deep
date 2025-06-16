# PER_numba.py
import numpy as np
from numba import jit

# Note: The @jit decorator is applied to functions with loops and numerical
# operations that are hard to vectorize with NumPy alone.

@jit(nopython=True)
def _propagate_numba(tree, idxs, changes):
    """
    Propagates changes in priority up the tree.
    This Numba-jitted function replaces the recursive Python method.
    """
    # Numba works best with explicit loops.
    for i in range(len(idxs)):
        idx = idxs[i]
        change = changes[i]
        
        parent = (idx - 1) // 2
        while True:
            tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

@jit(nopython=True)
def _retrieve_numba(tree, capacity, s):
    """
    Finds the leaf indices for a batch of priority values s.
    This is a heavily optimized version using Numba to compile the loops.
    """
    indices = np.zeros_like(s, dtype=np.int32)
    
    for i in range(len(s)):
        idx = 0
        val = s[i]
        while idx < capacity - 1:
            left_child_idx = 2 * idx + 1
            right_child_idx = left_child_idx + 1
            
            if val <= tree[left_child_idx]:
                idx = left_child_idx
            else:
                val -= tree[left_child_idx]
                idx = right_child_idx
        indices[i] = idx
    return indices

class SumTree:
    """
    A SumTree implementation optimized with Numba for critical path methods.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idxs, changes):
        # Call the external, Numba-jitted function
        _propagate_numba(self.tree, idxs, changes)

    def _retrieve(self, s):
        # Call the external, Numba-jitted function
        return _retrieve_numba(self.tree, self.capacity, s)

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        """Adds a new experience with a given priority."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """Updates the priority of an existing experience."""
        # This can be a single value or an array for batch updates
        if isinstance(tree_idx, int):
            tree_idx = np.array([tree_idx])
            priority = np.array([priority])
            
        changes = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, changes)

    def get(self, s):
        """
        Retrieves samples based on priority values s.
        """
        idxs = self._retrieve(s)
        data_idxs = idxs - self.capacity + 1
        return (idxs, self.tree[idxs], self.data[data_idxs])

class PERBuffer:
    """ Prioritized Experience Replay Buffer using a Numba-optimized SumTree. """
    e = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def store(self, experience):
        """Stores an experience with maximum priority."""
        max_priority = np.max(self.tree.tree[-self.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, experience)

    def sample(self, n):
        """Samples a batch of n experiences from the buffer."""
        if self.tree.n_entries == 0:
             return [], [], np.array([])
        
        total_p = self.tree.total()
        segment = total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        s = np.random.uniform(segment * np.arange(n), segment * np.arange(1, n + 1))
        
        idxs, priorities, batch = self.tree.get(s)
        
        sampling_probabilities = priorities / total_p
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, np.array(is_weights, dtype=np.float32)

    def batch_update(self, tree_idxs, abs_errors):
        """Updates the priorities of the sampled experiences."""
        priorities = np.power(abs_errors + self.e, self.alpha)
        self.tree.update(tree_idxs, priorities)

    def __len__(self):
        return self.tree.n_entries