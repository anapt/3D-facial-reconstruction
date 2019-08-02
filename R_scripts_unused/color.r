setwd("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads")

library(readobj)
library(rgl)
library(mdatools)
library(readobj)
library(rgl)
library(png)

average = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average_1.obj", materialspath = NULL, convert.rgl = FALSE)
average = sample_vertices(average)
averagePNG = readPNG("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average.png")
#when plotting the whole model use the following instruction
average$shapes$all_triangles$texcoords = array(average$shapes$all_triangles$texcoords, dim=c(2, 75870))

# texture coord (v) fix
average$shapes$all_triangles$texcoords[2, ] = average$shapes$all_triangles$texcoords[2,] - 511

# texel_coord = uv_coord * [width, height]
scale_c = length(averagePNG[,1,1])
c1 = c(floor(average$shapes$all_triangles$texcoords[1,]*scale_c)) # u 
c2 = c(floor(scale_c - average$shapes$all_triangles$texcoords[2,]*scale_c)) # v

# color
color = matrix(0,3,length(average$shapes$all_triangles$positions[1,]))
for (i in c(1:length(average$shapes$all_triangles$positions[1,]))){
  color[,i] = averagePNG[c2[i], c1[i],]
}

# plot
plot3d(average$shapes$all_triangles$positions[1,], 
       average$shapes$all_triangles$positions[2,], average$shapes$all_triangles$positions[3,], col = rgb(t(color)))

