setwd("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads")

library(readobj)
library(rgl)
library(mdatools)
library(readobj)
library(rgl)
library(png)

average = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average_1.obj", materialspath = NULL, convert.rgl = FALSE)
average = sample_vertices(average)
# scale
average$shapes$all_triangles$texcoords[2, ] = average$shapes$all_triangles$texcoords[2,]/1000

c1 = c(floor(average$shapes$all_triangles$texcoords[1,]*256)) # u 
c2 = c(floor(average$shapes$all_triangles$texcoords[2,]*256)) # v
# texel_coord = uv_coord * [width, height]

setwd("D:/[THESIS] Data/Cyberware TM/faces/FacesDatabase/faces/png")

files = grep("_0r.png", list.files(path = "."), value = TRUE)

 
idata <- data.frame(matrix(0,1,200))
data = matrix(0,3,length(average$shapes$all_triangles$positions[1,]))

for (i in c(1:length(files))){
  
  f = paste("D:/[THESIS] Data/Cyberware TM/faces/FacesDatabase/faces/png/",files[i], sep = "")
  iPNG = readPNG(f)
  
  for (j in c(1:length(average$shapes$all_triangles$positions[1,]))){
    data[,j] = iPNG[c2[j], c1[j],]
  }
  
  idata[i] <- array(data, dim= c(1, 72000))
}


pca_model <- prcomp(idata, center = F, scale = F)

#eigenvalues = pca_model$sdev^2
#eigenvectors = pca_model$rotation
#barplot(eigenvalues / sum(eigenvalues))

idata_pc <- as.data.frame(predict(pca_model, idata)[, 1:80])