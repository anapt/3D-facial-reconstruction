setwd('~/Documents/Thesis - data')

library(rhdf5)
library(rgl)

h5ls("data-raw/model2017-1_bfm_nomouth.h5")

pca_shape <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/model")

pca_color <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/color/model")

pca_expression <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/expression/model")

shape = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/representer")
color = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/color/representer")

# plot
pca_color$mean = array(pca_color$mean, dim = c(3, length(pca_color$mean)/3))
plot3d(color[["points"]], col = rgb(t(pca_color$mean)))
