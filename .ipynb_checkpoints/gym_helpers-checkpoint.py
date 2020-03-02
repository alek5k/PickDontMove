# Takes a sample from a space (eg. action or observation space) and returns it as a flattened list
def flatten_space_sample(space_sample):
    import numpy as np
    output = []
    for entry in space_sample:
        if isinstance(entry, np.ndarray):
            for array_item in entry:
                output.append(array_item)
        else:
            output.append(entry)
    return output
    # for key, value in space_sample.items():
    #     import numpy as np
    #     output = []
    #     if isinstance(value, np.ndarray):
    #         output.extend(value)
    #     else:
    #         output.append(entry)
    #     return output

