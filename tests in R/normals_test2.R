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
triangles = cells

count = zeros(1,53149)
count = t(count)
normals = zeros(3, 53149)
e = 0.1

for (i in c(1:105694)){
  
  idx = triangles[,i]
  
  A = vertices[, idx[1]+1]
  B = vertices[, idx[2]+1]
  C = vertices[, idx[3]+1]

  n = cross(B-A, C-A)
  area = ((n[1]^2+n[2]^2+n[3]^2)^0.5)/2

  if (dot(normals[,idx[1]+1], n) > e ||  dot(normals[,idx[1]+1], n) == 0){
    normals[,idx[1]+1] = area * n + normals[,idx[1]+1]
    count[idx[1]+1] = count[idx[1]+1] + 1
  }
    
  if (dot(normals[,idx[2]+1], n) > e ||  dot(normals[,idx[2]+1], n) == 0){
    normals[,idx[2]+1] = area * n + normals[,idx[2]+1]
    count[idx[2]+1] = count[idx[2]+1] + 1
  }
  if (dot(normals[,idx[3]+1], n) > e ||  dot(normals[,idx[3]+1], n) == 0){
    normals[,idx[3]+1] = area * n + normals[,idx[3]+1]
    count[idx[3]+1] = count[idx[3]+1] + 1
  }
  
  
  
  
  
}

for (i in c(1:53149)){
  normals[,i] = normals[,i] / count[i]
}

plot3d(t(normals))

normals = normalize(normals, method = 'range', range = c(-1,1))
