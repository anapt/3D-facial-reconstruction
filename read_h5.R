setwd('~/Documents/Thesis - data')

library(rhdf5)
library(rgl)

# h5ls("data-raw/model2017-1_bfm_nomouth.h5")

# N = 53149
# Es [3N 80]
#pca_shape <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/model")
Es <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/model/pcaBasis")
Es <- t(Es[1:80,])
# Er [3N 80]
#pca_color <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/color/model")
Er <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/color/model/pcaBasis")
Er <- t(Er[1:80,])
# Ee [3N 64]
#pca_expression <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/expression/model")
Ee <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/expression/model/pcaBasis")
Ee <- t(Ee[1:64,])

As = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/model/mean")
As = t(t(As))
Ar = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/color/model/mean") # 159447
Ar = t(t(Ar))

# plot
color = array(Ar, dim = c(3, length(Ar)/3))
points_av = array(As, dim = c(3, length(As)/3))
plot3d(points_av[1,], points_av[2,], points_av[3,], col = rgb(t(color)))

# x = [a, d, b, T, t, g]

# sampling
# sample a from a normal distribution N(0,1)
a = rnorm(80, 0 , 1)
d = runif(64,-24,24)
d[1] = runif(1,-4.8,4.8)
b = rnorm(80, 0 , 1)

# eq 2
V = As + Es %*% a + Ee %*% d
V2 = array(V, dim = c(3, length(V)/3))
# eq 3
R = Ar + Er %*% b
R2 = array(R, dim = c(3, length(R)/3))
frame()
plot3d(V2[1,], V2[2,], V2[3,], col = rgb(t(color)))

a1 = rnorm(80, 0 , 1)

d1 = runif(64,-12,12)
d1[1] = runif(1,-4.8,4.8)
b1 = rnorm(80, 0 , 1)

# eq 2
V1 = As + Es %*% a1 + Ee %*% d1
V21 = array(V, dim = c(3, length(V)/3))
# eq 3
R1 = Ar + Er %*% b1
R21 = array(R, dim = c(3, length(R)/3))

