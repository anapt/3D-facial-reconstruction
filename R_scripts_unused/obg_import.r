library(readobj)
library(rgl)

setwd("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads")

average = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/average.obj", materialspath = NULL, convert.rgl = FALSE)
barbara = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/barbara.obj", materialspath = NULL, convert.rgl = FALSE)
isabelle = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/isabelle.obj", materialspath = NULL, convert.rgl = FALSE)
thomas = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/thomas.obj", materialspath = NULL, convert.rgl = FALSE)
volker = read.obj("D:/[THESIS] Data/Cyberware TM/heads/FacesDatabase/heads/volker.obj", materialspath = NULL, convert.rgl = FALSE)


rand = sample(ncol(average$shapes$all_triangles$positions),24000)
# vertices coords
average$shapes$all_triangles$positions <- average$shapes$all_triangles$positions[,rand]
# vertices normals
average$shapes$all_triangles$normals <- average$shapes$all_triangles$normals[,rand]
# texture coordinates
average$shapes$all_triangles$texcoords = array(average$shapes$all_triangles$texcoords, dim=c(2, 75870))
average$shapes$all_triangles$texcoords <- average$shapes$all_triangles$texcoords[,rand]

v <- c()
for (i in 1:150750){
  if (average$shapes$all_triangles$indices[1, i] %in% rand 
      && (average$shapes$all_triangles$indices[2, i] %in% rand) 
      && (average$shapes$all_triangles$indices[3, i] %in% rand)){
    v <- c(v, average$shapes$all_triangles$indices[,i])
    v
  }
    
}

v <- array(v, dim=c(3,4720))
#plot3d(savg[1,], savg[2,], savg[3,])