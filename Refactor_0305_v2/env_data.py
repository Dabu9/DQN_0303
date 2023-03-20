import numpy as np
obstacle = [[[1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]],

            [[1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]],
]

print([1, 2]+np.array([obstacle[0][0][0], obstacle[0][0][1]]))