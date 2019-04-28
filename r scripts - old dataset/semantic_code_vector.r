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
averagePNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average.png")

# average face shape dim = 3N
As = average$shapes$all_triangles$positions
# stack coordinates as X1 Y1 Z1 X2 Y2 .... YN ZN
As = array(As, dim=c(1, length(As)))

# TODO find PCA basis Es 3Nx80 -> encodes the modes with the highest shape variation
# TODO find PCA basis Ee 3Nx64 -> encodes the modes with the highest expression variation

#average skin reflectance dim = 3N
# texture coord (v) fix
average$shapes$all_triangles$texcoords[2, ] = average$shapes$all_triangles$texcoords[2,] - 511

# texel_coord = uv_coord * [width, height]
scale_c = length(averagePNG[,1,1])
c1 = c(floor(average$shapes$all_triangles$texcoords[1,]*scale_c)) # u 
c2 = c(floor(scale_c - average$shapes$all_triangles$texcoords[2,]*scale_c)) # v
# skin reflectance
Ar = matrix(0,3,length(average$shapes$all_triangles$positions[1,]))
for (i in c(1:length(average$shapes$all_triangles$positions[1,]))){
  Ar[,i] = averagePNG[c2[i], c1[i],]
}
# stack skin reflectance as R1 G1 B1 R2 G2 .... GN BN
Ar = array(Ar, dim=c(1, length(Ar)))

# TODO find orthogonal PCA basis Er 3Nx80 -> captures the modes of highest reflectance variation