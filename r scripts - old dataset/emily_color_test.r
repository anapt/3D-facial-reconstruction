library(readobj)
library(rgl)
library(mdatools)
library(readobj)
library(rgl)
library(png)

setwd("D:/[THESIS] Data/DigitalEmily/Emily_2_1_OBJ")
emily = read.obj("D:/[THESIS] Data/DigitalEmily/Emily_2_1_OBJ/Emily_2_1.obj", materialspath = NULL, convert.rgl = FALSE)

eye_inner_PNG_2 = readPNG("D:/[THESIS] Data/DigitalEmily/Emily_2_1_Textures/Textures/Color_raw/Eye_Inner_Iris_Bump_01.png")
#when plotting the whole model use the following instruction
emily$shapes$Eye_Inner_01$texcoords = array(emily$shapes$Eye_Inner_01$texcoords, dim = c(2,20488))


c1 = c(floor(4096 - emily$shapes$Eye_Inner_01$texcoords[1,]*4096))
c2 = c(floor(emily$shapes$Eye_Inner_01$texcoords[2,]*4096))
coords = matrix(0,2,length(emily$shapes$Eye_Inner_01$positions[1,]))
coords[1,]=c1
coords[2,]=c2

color = matrix(0,4,length(emily$shapes$Eye_Inner_01$positions[1,]))
for (i in c(1:length(emily$shapes$Eye_Inner_01$positions[1,]))){
  color[,i] = eye_inner_PNG_1[c2[i], c1[i],]
}
plot3d(t(emily$shapes$Eye_Inner_01$positions), col = rgb(t(color)))
