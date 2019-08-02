library(rgl)
library(plyr)
library(rhdf5)
library(pracma)
library(BBmisc)

setwd('~/Documents/Thesis - data')
vertices = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/model/mean")
vertices = array(vertices, dim = c(3, length(vertices)/3))
cells = h5read("data-raw/model2017-1_bfm_nomouth.h5", "/shape/representer/cells")
cells = t(cells)

nor = zeros(3, 53149)

cells = cells[,1:1000]
vertices = vertices[,1:2000]
#struct Vertex { Position p,   Normal n }
#VertexList v
#epsilon e

#for each vertex i in VertexList v
#  n ← Zero Vector
#  m ← Normalize(Normal(v, i%3))
#  for each triangle j that shares i th vertex
#    q ← Normalize(Normal(v, j))
#    w ← Area(v, j)
#    if DotProduct(q, m) > e
#      n ← n + w*q
#    end if
#  end for
#  v[i].n ← Normalize(n)
#end for

e = 0.1
for (i in c(1:size(vertices)[2]-1)){
  n = zeros(3,1)
  m = ones(3,1)
  #m = normalize(normal_one(vertices, i/3))
  for (j in c(1:size(cells)[2])){
    if (i %in% cells[,j]){
      #print(i)
      #print(cells[,j])
      q = normalize(normal(vertices,cells[,j]))
      w = area(vertices, cells[,j])
      #print("here")
      #print(q)
      #print(m)
      #print(dot(q,m))
      #if (dot(q,m) > e){
        n = n + w*q
      #}
    }
  }
  nor[,i] = n
  print(nor[,i])
}



area <- function(vertices, j){
  #print(j)
  A <- vertices[,j[1]+1]
  B <- vertices[,j[2]+1]
  C <- vertices[,j[3]+1]

  product = cross(B-A, C-A)
  area <- product[1]^2 + product[2]^2 + product[3]^2
  area <- sqrt(area) / 2

  #print(area)
}

normal_one <- function(vertices, j){
  
  j = j*3
  #print(j)
  A <- vertices[,j+1]
  B <- vertices[,j+2]
  C <- vertices[,j+3]
  
  normal <- cross(B-A, C-A)
  #print(normal)
  
}

normal <- function(vertices, j){
  #print(j)
  A <- vertices[,j[1]+1]
  B <- vertices[,j[2]+1]
  C <- vertices[,j[3]+1]
  
  normal <- cross(B-A, C-A)

}


plot3d(t(nor))

for (i in c(1:size(triangles)[2])){
  
  #vertices V[triangles[,1]+1]
  A = vertices[,triangles[,i]+1][,1]
  B = vertices[,triangles[,i]+1][,2]
  C = vertices[,triangles[,i]+1][,3]
  p = cross(B-A, C-A)
  
  normals[,triangles[,i]+1][,1] = normals[,triangles[,i]+1][,1]+ p
  normals[,triangles[,i]+1][,2] = normals[,triangles[,i]+1][,2]+p
  normals[,triangles[,i]+1][,3] = normals[,triangles[,i]+1][,3]+p
  
}
num_of_ap = table(triangles)
for (i in c(1:53149)){
  normals[,i] = normals[,i]/num_of_ap[i]
}