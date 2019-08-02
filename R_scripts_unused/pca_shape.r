library(readobj)
library(rgl)
library(png)

sample_vertices <- function(n){
  
  ncolumns = ncol(n$shapes$all_triangles$positions)
  
  rand = sample(ncol(n$shapes$all_triangles$positions),24000)
  # vertices coords
  n$shapes$all_triangles$positions <- n$shapes$all_triangles$positions[,rand]
  # vertices normals
  n$shapes$all_triangles$normals <- n$shapes$all_triangles$normals[,rand]
  # texture coordinates
  n$shapes$all_triangles$texcoords = array(n$shapes$all_triangles$texcoords, dim=c(2, ncolumns))
  n$shapes$all_triangles$texcoords <- n$shapes$all_triangles$texcoords[,rand]
  
  return(n)
  
}

setwd("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads")

average = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average_1.obj", materialspath = NULL, convert.rgl = FALSE)
average = sample_vertices(average)
#average$shapes$all_triangles$texcoords = array(average$shapes$all_triangles$texcoords, dim=c(2, length(average$shapes$all_triangles$positions[1,])))
averagePNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average.png")

barbara = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/barbara.obj", materialspath = NULL, convert.rgl = FALSE)
barbara = sample_vertices(barbara)
#barbara$shapes$all_triangles$texcoords = array(barbara$shapes$all_triangles$texcoords, dim=c(2, length(barbara$shapes$all_triangles$positions[1,])))
barbaraPNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/barbara.png")

isabelle = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/isabelle.obj", materialspath = NULL, convert.rgl = FALSE)
isabelle = sample_vertices(isabelle)
#isabelle$shapes$all_triangles$texcoords = array(isabelle$shapes$all_triangles$texcoords, dim=c(2, length(isabelle$shapes$all_triangles$positions[1,])))
isabellePNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/isabelle.png")

thomas = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/thomas.obj", materialspath = NULL, convert.rgl = FALSE)
thomas = sample_vertices(thomas)
#thomas$shapes$all_triangles$texcoords = array(thomas$shapes$all_triangles$texcoords, dim=c(2, length(thomas$shapes$all_triangles$positions[1,])))
thomasPNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/thomas.png")

volker = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/volker.obj", materialspath = NULL, convert.rgl = FALSE)
volker = sample_vertices(volker)
#volker$shapes$all_triangles$texcoords = array(volker$shapes$all_triangles$texcoords, dim=c(2, length(volker$shapes$all_triangles$positions[1,])))
volkerPNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/volker.png")


# average face shape dim = 3N
#As = average$shapes$all_triangles$positions
# stack coordinates as X1 Y1 Z1 X2 Y2 .... YN ZN
#As = array(As, dim=c(1, length(As)))

# TODO find PCA basis Es 3Nx80 -> encodes the modes with the highest shape variation
idata <- array(0, dim =c(72000,5))
idata[,1] <- array(average$shapes$all_triangles$positions, dim= c(1, 72000))
idata[,2] <- array(barbara$shapes$all_triangles$positions, dim= c(1, 72000))
idata[,3] <- array(isabelle$shapes$all_triangles$positions, dim= c(1, 72000))
idata[,4] <- array(thomas$shapes$all_triangles$positions, dim= c(1, 72000))
idata[,5] <- array(volker$shapes$all_triangles$positions, dim= c(1, 72000))


pca_model <- prcomp(idata, center = T, scale. = T)
eigenvalues = pca_model$sdev^2
eigenvectors = pca_model$rotation
barplot(eigenvalues / sum(eigenvalues))
