import pathlib
import numpy as np
from FaceNet3D import FaceNet3D as Helpers
from SemanticCodeVector import SemanticCodeVector
helper = Helpers()

sem_root = pathlib.Path(helper.sem_root + 'training/')

all_vector_paths = list(sem_root.glob('*'))
all_vector_paths = [str(path) for path in all_vector_paths]

all_vector_paths = np.random.choice(all_vector_paths, 4)
# print(a)

vectors = np.zeros((helper.scv_length, len(all_vector_paths)))
for n, path in enumerate(all_vector_paths):
    v = np.loadtxt(path)
    vectors[:, n] = np.asarray(v)

# print(vectors)

sum_shape = 0.0
sum_expression = 0.0
sum_color = 0.0
sum_rotation = 0.0
shape_std, color_std, expression_std = SemanticCodeVector().get_bases_std()

for i in range(0, vectors.shape[1]):
    x = helper.vector2dict(vectors[:, i])
    # print(i)

    square = shape_std * abs(np.power(x['shape'], 1))
    sum_shape += np.sum(square)
    # print(sum_shape)

    square = expression_std * abs(np.power(x['expression'], 1))
    sum_expression += np.sum(square)
    # print(sum_expression)

    square = color_std * abs(np.power(x['color'], 1))
    sum_color += np.sum(square)
    # print(sum_reflectance)

    square = abs(np.power(x['rotation'], 1))
    sum_rotation += np.sum(square)
    # print(sum_rotation)


avg_shape = (sum_shape/32)
avg_expression = (sum_expression/32)
avg_color = (sum_color/32)
avg_rotation = (sum_rotation/32)

constant = 1000
w_shape = constant / avg_shape                      # 1.57      1.46    1.39    1.6
w_expression = constant / avg_expression            # 2.33      2.19    2.29
w_color = constant / avg_color                      # 48.71     45.73   50.04
w_rotation = constant / avg_rotation                # 5454.13   6307    5823

print("Weights: \nshape: {}\nexpression: {}\ncolor: {}\nrotation: {}"
      .format(w_shape, w_expression, w_color, w_rotation))
