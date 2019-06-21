color <- function(r, n, g){
  
  color = zeros(3,1)
  for (i in c(1:9)){
    SH <- Hb(n[1], n[2], n[3], i)
    #print(SH)
    color = color + g[,i]*SH
  }
  #print(r)
  #print(color)
  color = r*color
  
}

#For each vertex
#vertex.n := (0, 0, 0)

#For each triangle ABC
#// compute the cross product and add it to each vertex
#p := cross(B-A, C-A)
#A.n += p
#B.n += p
#C.n += p

#For each vertex
#vertex.n := normalize(vertex.n)


normals <- function(cells, vertices){
  # cells     size  1:3, ....
  # vertices        1:3, ....
  normals = zeros(3, 53149)
  for (i in c(1:size(cells)[2])){
    #vertices V[cells[,1]+1]
    A = vertices[,cells[,i]+1][,1]
    B = vertices[,cells[,i]+1][,2]
    C = vertices[,cells[,i]+1][,3]
    p = cross(B-A, C-A)
    
    normals[,cells[,i]+1][,1] = normals[,cells[,i]+1][,1]+ p
    normals[,cells[,i]+1][,2] = normals[,cells[,i]+1][,2]+p
    normals[,cells[,i]+1][,3] = normals[,cells[,i]+1][,3]+p
    
  }
  normals = normalize(normals, method = "range", range = c(-1,1))

}
