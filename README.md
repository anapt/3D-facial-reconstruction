# Thesis

## Abstract
The present work implements an automatic system for coding and reconstructing 3D faces from low resolution RGB images by utilizing machine learning algorithms.
Given a 3D morphable model, different faces are represented as a vector of variables ("code vector") which describe the shape, expression and color of the face. The multiplication of these parameter vectors with the PCA bases provided by the morphable model results in the 3D coordinates of the reconstructed face.
As part of this work, an algorithm for the creation of two-dimensional synthetic faces solely from the information captured by the code vector was developed. The synthetic faces were used to train the neural network that acts as the encoding phase of the auto-encoding system and bootstrapping techniques were used to generalize the network to real-world facial images.

The outcome of this work is not only proof of the potential of 3D facial reconstruction from RGB images, but also the ability to exploit the 3D face by changing its expression, color or lighting. In the context of said exploitation, a neural network was implemented to identify the facial expression from the information encoded in the code vector.
The problem tackled by the present work has until now been solved by the use of iterative algorithms based on the linear combination of existing prototype samples, which require a large amount of data from three-dimensional scans. Here, an attempt is made to solve this problem purely with machine learning and synthetic data.

## Dependencies
```
pip install -r requirements.txt
```

## Bugs & Request
Please report bugs and request features using the [Issue Tracker](https://github.com/anapt/thesis/issues).