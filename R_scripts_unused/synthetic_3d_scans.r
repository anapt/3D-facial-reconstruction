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
#average$shapes$all_triangles$texcoords = array(average$shapes$all_triangles$texcoords, dim=c(2, length(average$shapes$all_triangles$positions[1,])))
average = sample_vertices(average)
averagePNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average.png")

barbara = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/barbara.obj", materialspath = NULL, convert.rgl = FALSE)
#barbara$shapes$all_triangles$texcoords = array(barbara$shapes$all_triangles$texcoords, dim=c(2, length(barbara$shapes$all_triangles$positions[1,])))
barbara = sample_vertices(barbara)
barbaraPNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/barbara.png")

### AVERAGE
## shape vector
shape_avg = average$shapes$all_triangles$positions
shape_avg = array(shape_avg, dim=c(1, length(shape_avg)))

## texture vector
# texture coord (v) fix
average$shapes$all_triangles$texcoords[2, ] = average$shapes$all_triangles$texcoords[2,] - 511
# texel_coord = uv_coord * [width, height]
scale_c = length(averagePNG[,1,1])
c1 = c(floor(average$shapes$all_triangles$texcoords[1,]*scale_c)) # u 
c2 = c(floor(scale_c - average$shapes$all_triangles$texcoords[2,]*scale_c)) # v
# color
texture_avg = matrix(0,3,length(average$shapes$all_triangles$positions[1,]))
for (i in c(1:length(average$shapes$all_triangles$positions[1,]))){
  texture_avg[,i] = averagePNG[c2[i], c1[i],]
}
texture_avg = array(texture_avg, dim=c(1, length(texture_avg)))


### BARBARA
## shape vector
shape_bar = barbara$shapes$all_triangles$positions
shape_bar = array(shape_bar, dim=c(1, length(shape_bar)))

## texture vector
# texture coord (v) fix
barbara$shapes$all_triangles$texcoords[2, ] = barbara$shapes$all_triangles$texcoords[2,] - 511
# texel_coord = uv_coord * [width, height]
scale_c = length(barbaraPNG[,1,1])
c1 = c(floor(barbara$shapes$all_triangles$texcoords[1,]*scale_c)) # u 
c2 = c(floor(scale_c - barbara$shapes$all_triangles$texcoords[2,]*scale_c)) # v
# color
texture_bar = matrix(0,3,length(barbara$shapes$all_triangles$positions[1,]))
for (i in c(1:length(barbara$shapes$all_triangles$positions[1,]))){
  texture_bar[,i] = barbaraPNG[c2[i], c1[i],]
}
texture_bar = array(texture_bar, dim=c(1, length(texture_bar)))

## CLEANUP
rm(c1,c2,i,scale_c,average, averagePNG, barbara, barbaraPNG)
rm(sample_vertices)
