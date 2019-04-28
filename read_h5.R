setwd('~/Documents/Thesis - data')

library(rhdf5)

h5ls("data-raw/model2017-1_bfm_nomouth.h5")

pca_shape <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/model")

pca_color <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/color/model")

pca_expression <- h5read("data-raw/model2017-1_bfm_nomouth.h5", "/expression/model")
