# GDAL command example:
# gdal_rasterize -l Roads -a Shape_STLe -tr 30.0 30.0 -a_nodata 0.0 -te 1879247.9322 3937927.1331 1892280.2214 3946916.313 -ot Float32 -of GTiff /Users/arqam/abm/UNSW/Roads.shp /Users/arqam/abm/final_rds1.tif
library(raster)
library(tidyverse)


# Set resolution: 1(30x30) 2(20x20) 3(10x10)
res <- 3

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#import TIFF rasters
rvr_raster <- raster(paste( "final_rvr", res, ".tif", sep=""))
bld_raster <- raster(paste( "final_bld", res, ".tif", sep=""))
rds_raster <- raster(paste( "final_rds", res, ".tif", sep =""))


#visualise
plot(rvr_raster)
plot(bld_raster, add=T)
plot(rds_raster, add =T)



# convert river data to matrix
rvr_mtx <- as.matrix(rvr_raster)

rvr_mtx <- ifelse(is.na(rvr_mtx)==T, 0, rvr_mtx)
rvr_mtx <- ifelse(rvr_mtx !=0, 1, rvr_mtx)

sum(rvr_mtx)

# buildings data
bld_mtx <- as.matrix(bld_raster)

bld_mtx <- ifelse(is.na(bld_mtx)==T, 0, bld_mtx)
bld_mtx <- ifelse(bld_mtx !=0, 2, bld_mtx)

sum(bld_mtx)

# merging into single matrix
dat <- rvr_mtx + bld_mtx
dat <- ifelse(dat == 3, 1, dat) # if river and building at some grid point, it is considered part of river

# road data

rds_mtx <- as.matrix(rds_raster)

rds_mtx <- ifelse(is.na(rds_mtx)==T, 0, rds_mtx)
rds_mtx <- ifelse(rds_mtx !=0, 3, rds_mtx)

sum(rds_mtx)

# merging

dat <- dat+ rds_mtx

dat <- ifelse(dat == 4, 1, dat)
dat <- ifelse(dat == 5, 2, dat)

dim(dat)

sum(rvr_mtx) + sum(bld_mtx) +sum(rds_mtx)
# should be greater than
sum(dat)

txt <- matrix("_", dim(dat)[1],dim(dat)[2])
dim(txt)

for(i in 2:dim(dat)[1]-1){
  for(j in 1:dim(dat)[2]){
    if(dat[i,j] == 1) txt[i,j] <- "R"
    else if(dat[i,j] == 2) txt[i,j] <- "B"
    #river boundary
    if( (dat[i-1,j] == 0) && (dat[i,j] == 1)) txt[i,j] <- "W"
    else if( (dat[i+1,j] == 0) && (dat[i,j] == 1)) txt[i,j] <- "W"
  }
}
any(is.na(dat))

write.table(txt, file=paste("grid",res,".txt", sep =""), row.names=FALSE, col.names=FALSE, quote=F)
