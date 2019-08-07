import pathlib
import numpy as np

sem_root = './DATASET/semantic/training/'

sem_root = pathlib.Path(sem_root)

all_vector_paths = list(sem_root.glob('*'))
all_vector_paths = [str(path) for path in all_vector_paths]

all_vector_paths = np.random.choice(all_vector_paths, 32)
# print(a)

vectors = np.zeros((257, len(all_vector_paths)))
for n, path in enumerate(all_vector_paths):
    v = np.loadtxt(path)
    vectors[:, n] = np.asarray(v)

# print(vectors)


def vector2dict(vector):
    """
    Method that transforms (257,) nd.array to dictionary

    :param vector: <class 'numpy.ndarray'> with shape (257, ) : semantic code vector
    :return:
    dictionary with keys    shape           (80,)
                            expression      (64,)
                            reflectance     (80,)
                            rotation        (3,)
                            translation     (3,)
                            illumination    (27,)
    """
    if isinstance(vector, dict):
        return vector
    else:
        x = {
            "shape": np.squeeze(vector[0:80, ]),
            "expression": np.squeeze(vector[80:144, ]),
            "reflectance": np.squeeze(vector[144:224, ]),
            "rotation": np.squeeze(vector[224:227, ]),
            "translation": np.squeeze(vector[227:230, ]),
            "illumination": np.squeeze(vector[230:257, ])
        }
        return x



sum_shape = 0.0
sum_expression = 0.0
sum_reflectance = 0.0
sum_rotation = 0.0
sum_translation = 0.0
sum_illumination = 0.0

for i in range(0, vectors.shape[1]):
    x = vector2dict(vectors[:, i])
    # print(i)

    square = np.power(x['shape'], 2)
    sum_shape += np.sum(square)
    # print(sum_shape)

    square = np.power(x['expression'], 2)
    sum_expression += np.sum(square)
    # print(sum_expression)

    square = np.power(x['reflectance'], 2)
    sum_reflectance += np.sum(square)
    # print(sum_reflectance)

    square = np.power(x['rotation'], 2)
    sum_rotation += np.sum(square)
    # print(sum_rotation)

    square = np.power(x['translation'], 2)
    sum_translation += np.sum(square)
    # print(sum_translation)

    square = np.power(x['illumination'], 2)
    sum_illumination += np.sum(square)
    # print(sum_illumination)

avg_shape = (sum_shape/32)
avg_expression = (sum_expression/32)
avg_reflectance = (sum_reflectance/32)
avg_rotation = (sum_rotation/32)
avg_translation = (sum_translation/32)
avg_illumination = (sum_illumination/32)

constant = 10000
w_shape = constant / avg_shape                      # 125
w_expression = constant / avg_expression            # 1
w_reflectance = constant / avg_reflectance          # 5500
w_rotation = constant / avg_rotation                # 55
w_translation = constant / avg_translation          # 750
w_illumination = constant / avg_illumination        # 3300

print("Weights: \nshape: {}\nexpression: {}\nreflectance: {}\nrotation: {}\ntranslation: {}\nillumination: {}\n"
      .format(w_shape, w_expression, w_reflectance, w_rotation, w_translation, w_illumination))
