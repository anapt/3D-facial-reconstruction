setwd("D:/[THESIS] Data/Cyberware TM/faces/FacesDatabase/faces/png")

library(EBImage)
library(png)

files = grep("MPIf001_", list.files(path = "."), value = TRUE)

PNG = array(0, dim=c(256,256,3))
for (i in c(1:length(files))){
  
  f = paste("D:/[THESIS] Data/Cyberware TM/faces/FacesDatabase/faces/png/",files[i], sep = "")
  iPNG = readPNG(f)
  PNG = PNG + iPNG
}
m = 
## matrices corresponding to red, green and blue color channels
r <- PNG[,,1]/7
g <- PNG[,,2]/7
b <- PNG[,,3]/7

## construct an color Image object 
img <- rgbImage(r, g, b)

display(img, method="raster", all=TRUE)


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