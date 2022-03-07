from typing import List, Tuple

import numpy as np
from numba import njit

__all__ = ["fast_greedy_bucket_elements"]


@njit(nogil=True)
def fast_greedy_bucket_elements(
    element_sizes: List[int], allowed_bucket_sizes: Tuple[int, ...]
) -> Tuple[List[int], List[List[int]]]:
    """
    This implementation uses a greedy approximation of the ILP and in some experiments has been shown to work quite well.

    ### ILP Definition

        y_j binary variable for j=1..k whether bucket j is used (contains any elements)
        L_j the size of the bucket

        x_ij binary variable for i=1..n and j=1..k whether the element i is in bucket j
        s_i  the size of element i

        Constraints:
          sum(x_ij for j in 1..k) == 1 for all i  # Every element is assigned to exactly one bucket
          sum(x_ij * s_i for i in 1..k) <= y_j * L_j  for all j  # For all used buckets, their size should not be exceeded

        Objective (minimize):
          sum(L_j**2 * y_j for j in 1..k) - sum(l_i**2 for all i)  # how much space is "wasted"

    ### ILP Code
        For reference, the following (sloow) CP-SAT implementation can compute the solution to this problem exactly:
        ```
        # Example Data
        bucket_sizes = [8] * 500 + [16] * 350 + [32] * 200 + [64] * 100 + [128] * 90 + [256] * 30 + [512] * 40 + [768] * 50 + [1024] * 50
        bucket_size_sq = [s ** 2 for s in bucket_sizes]

        element_len = [int(x) for x in np.clip(np.random.zipf(1.5, size=1000) + 2, 2, 1024)] # Min hyperedge size is 3
        element_len_sq = [x**2 for x in element_len]

        num_buckets, num_elements = len(bucket_sizes), len(element_len)

        ## The actual ILP
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()

        # Define_variables
        bucket_is_used = [model.NewIntVar(0, 1, f'y[{i}]') for i in range(num_buckets)]
        allocations = [[model.NewIntVar(0, 1, f'x[{i}, {j}]') for j in range(num_buckets)] for i in range(num_elements)]

        # Constraints

        ## Everything should be allocated once
        for i in range(num_elements):
            model.Add(cp_model.LinearExpr.Sum(allocations[i]) == 1)

        ## Each bucket cannot exceed its capacity
        for j in range(num_buckets):
            model.Add(cp_model.LinearExpr.Sum(allocations[i][j]*element_len[i] for i in range(num_elements)) <= bucket_is_used[j] * bucket_sizes[j])

        model.Minimize(cp_model.LinearExpr.ScalProd(bucket_is_used, bucket_size_sq) - sum(bucket_size_sq))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 240
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f'Total cost = {solver.ObjectiveValue()} (status: {status})')
            print('Problem solved in %f milliseconds' % solver.WallTime())
            print("bucket_is_used=", [solver.Value(b) for b in bucket_is_used])
            print("allocations=", [[solver.Value(allocations[i][j]) for j in range(num_buckets)].index(1) for i in range(num_elements)])
        else:
            print('No solution found.')
        ```
    """
    sorted_idxs = np.argsort(element_sizes)[::-1]

    bucket_sizes: List[int] = []
    buckets_remaining_sizes: List[Tuple[int, int]] = []
    buckets_elements: List[List[int]] = []
    # Iterate over elements, starting with largest ones
    for element_idx in sorted_idxs:
        element_size = element_sizes[element_idx]
        # Try to find an existing bucket that fits:
        chosen_bucket_idx = None
        for remaining_bucket_idx, (bucket_idx, remaining_bucket_size) in enumerate(buckets_remaining_sizes):
            if remaining_bucket_size >= element_size:
                chosen_bucket_idx = bucket_idx
                break

        if chosen_bucket_idx is not None:
            # Use that remaining_bucket_size is set to the last value it had in the loop:
            remaining_bucket_size -= element_size
            buckets_elements[chosen_bucket_idx].append(element_idx)
            if remaining_bucket_size < 3:  # every hyperedge has at least size 3 (2 args + type)
                del buckets_remaining_sizes[remaining_bucket_idx]
            else:
                buckets_remaining_sizes[remaining_bucket_idx] = (chosen_bucket_idx, remaining_bucket_size)
        else:
            # Find smallest bucket size that would take this element (uses that
            # bucket sizes are sorted):
            for bsize in allowed_bucket_sizes:
                if element_size <= bsize:
                    new_bucket_size = bsize
                    break
            new_bucket_idx = len(bucket_sizes)
            bucket_sizes.append(new_bucket_size)
            buckets_remaining_sizes.append((new_bucket_idx, new_bucket_size - element_size))
            buckets_elements.append([element_idx])

    return bucket_sizes, buckets_elements
