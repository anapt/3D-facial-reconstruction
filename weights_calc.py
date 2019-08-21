import pathlib
import numpy as np
from FaceNet3D import FaceNet3D as Helpers
from SemanticCodeVector import SemanticCodeVector
helper = Helpers()

sem_root = pathlib.Path(helper.sem_root + 'training/')

all_vector_paths = list(sem_root.glob('*'))
all_vector_paths = [str(path) for path in all_vector_paths]

# all_vector_paths = np.random.choice(all_vector_paths, 32)
# print(a)
all_vector_paths = all_vector_paths[0:1]
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

    square = shape_std * abs(np.power(x['shape'], 2))
    sum_shape += np.sum(square)
    # print(sum_shape)

    square = expression_std * abs(np.power(x['expression'], 2))
    sum_expression += np.sum(square)
    # print(sum_expression)

    square = color_std * abs(np.power(x['color'], 2))
    sum_color += np.sum(square)
    # print(sum_reflectance)

    square = abs(np.power(x['rotation'], 2))
    sum_rotation += np.sum(square)
    # print(sum_rotation)

# for i in range(0, vectors.shape[1]):
#     x = helper.vector2dict(vectors[:, i])
#     # print(i)
#
#     square = abs(np.power(x['shape'], 2))
#     sum_shape += np.sum(square)
#     # print(sum_shape)
#
#     square = abs(np.power(x['expression'], 2))
#     sum_expression += np.sum(square)
#     # print(sum_expression)
#
#     square = abs(np.power(x['color'], 2))
#     sum_color += np.sum(square)
#     # print(sum_reflectance)
#
#     square = abs(np.power(x['rotation'], 2))
#     # print(square)
#     sum_rotation += np.sum(square)
#     # print(sum_rotation)


# avg_shape = (sum_shape/32)
# avg_expression = (sum_expression/32)
# avg_color = (sum_color/32)
# avg_rotation = (sum_rotation/32)

# avg_shape = (sum_shape/vectors.shape[1])
# avg_expression = (sum_expression/vectors.shape[1])
# avg_color = (sum_color/vectors.shape[1])
# avg_rotation = (sum_rotation/vectors.shape[1])
# print("avg_rot", avg_rotation)

avg_shape = (sum_shape)
avg_expression = (sum_expression)
avg_color = (sum_color)
avg_rotation = (sum_rotation)

constant = 1000
w_shape = constant / avg_shape
w_expression = constant / avg_expression
w_color = constant / avg_color
w_rotation = constant / avg_rotation

print("Weights: \nshape: {}\nexpression: {}\ncolor: {}\nrotation: {}"
      .format(w_shape, w_expression, w_color, w_rotation))
