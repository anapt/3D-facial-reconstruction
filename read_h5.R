setwd('~/Documents/Thesis - data')

library(rhdf5)
library(rgl)
library(pracma)
library(BBmisc)

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

# normals
cells = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/representer/cells")
cells = t(cells)

As = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/model/mean")
As = t(t(As))
Ar = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/color/model/mean") # 159447
Ar = t(t(Ar))

# plot
illumination1 = array(Ar, dim = c(3, length(Ar)/3))
points_av = array(As, dim = c(3, length(As)/3))
plot3d(points_av[1,], points_av[2,], points_av[3,], col = rgb(t(illumination1)))

# x = [a, d, b, T, t, g]

# sampling
# sample a from a normal distribution N(0,1)

a = rnorm(80, 0 , 1)      #shape
d = runif(64,-24,24)      #expression
d[1] = runif(1,-4.8,4.8)  
b = rnorm(80, 0 , 1)      #reflectance
T = runif(3, -40,40)      #rotation
T[3] = runif(1, -15,15)
g = runif(27, -0.2,0.2)   #illumination
g[1] = runif(1,0.6,1.2)
g = array(g, dim=c(3,9))


# eq 2
V = As + Es %*% a + Ee %*% d
V2d = array(V, dim = c(3, length(V)/3))
# eq 3
R = Ar + Er %*% b
R2d = array(R, dim = c(3, length(R)/3))

plot3d(V2d[1,], V2d[2,], V2d[3,], col = rgb(t(R2d)))

#------------------------------------------------------------------------------------------

# projection

N = normals(cells, V2d)

illumination = zeros(3, 53149)
for (i in c(1:size(V2d)[2])){
  illumination[,i] = color(R2d[,i], N[,i],g)
}
illumination = normalize(illumination, method = "range", range=c(0,1))

# plot
plot3d(V2d[1,], V2d[2,], V2d[3,], col = rgb(t(illumination)))

