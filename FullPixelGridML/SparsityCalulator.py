def generate_sparse_grid(x_size, y_size, s):
    """
    Generates a 1D array of flattened sparse grid coordinates.

    Parameters:
        x_size (int): X size.
        y_size (int): Y size.
        s (int): Sparseness level (step size between coordinates).

    Returns:
        List of coordinates representing flattened sparse grid coordinates.
    """
    # return [x * y_size + y for x in range(0, x_size, s) for y in range(0, y_size, s)]
    return [(x,y) for x in range(0, x_size, s) for y in range(0, y_size, s)]
# for s in range(1,25):
#     if (84,84) in generate_sparse_grid(85,85,s):
#         print(s)
#     else:
#         continue
print(generate_sparse_grid(100 ,100 ,14))
print(len(generate_sparse_grid(100,100,14)))