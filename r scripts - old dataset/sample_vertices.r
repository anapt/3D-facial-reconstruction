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
