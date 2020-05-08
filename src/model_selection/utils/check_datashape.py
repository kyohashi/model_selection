def check_array(array, ensure_2d=True):
    if ensure_2d:
        # If input is scalar raise error
        if array.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead:\narray={}.\n".format(array))
        # If input is 1D raise error
        if array.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n".format(array))
        # If input is more than 3D raise error
        if array.ndim >= 3:
            raise ValueError(
                "Expected 2D array, got more than 3D array instead:\narray={}.\n".format(array))
    return array
