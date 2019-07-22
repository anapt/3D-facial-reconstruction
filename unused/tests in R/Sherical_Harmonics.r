Hb <- function(x_coord, y_coord, z_coord, b ){
  
  v = c(x_coord, y_coord, z_coord);
  r = x_coord^2 + y_coord^2 + z_coord^2;
  if (b == 1){
    Hb = 0.75 * (35/pi)^(0.5) * (x_coord*y_coord*(x_coord^2 - y_coord^2))/(r^4);
  }
    else if (b == 2){
      Hb = 0.75 * (35/(2*pi))^(0.5) *(z_coord*y_coord*(3*x_coord^2 - y_coord^2))/(r^4);
    }
    
    else if (b == 3){
      Hb = 0.75 * (5/pi)^(0.5) * (x_coord*y_coord*(7*z_coord^2 - r^2))/(r^4);
    }
    
    else if (b == 4) {
      Hb = 0.75 * (5/(2*pi))^(0.5) * (y_coord*z_coord*(7*z_coord^2 - 3*r^2))/(r^4);
    }
    
    else if (b == 5){
      Hb = (3/16) * (1/pi)^(0.5) * (35*z_coord^2 - 30*z_coord^2*r^2 + 3*r^4)/(r^4);
    }
    
    else if (b == 6){
      Hb = 0.75 * (5/(2*pi))^(0.5) * (x_coord*z_coord*(7*z_coord^2 - 3*r^2))/(r^4);
    }
    
    else if (b == 7){
      Hb = (3/8) * (5/pi)^(0.5) * ((x_coord^2 - y_coord^2) *(7*z_coord^2 - r^2))/(r^4);
    }
    
    else if (b == 8){
      Hb = (3/4) * (35/(2*pi))^(0.5) * ((x_coord^2 - 3*y_coord^2)*x_coord*y_coord)/(r^4);
    }
    
    else if (b == 9){
      Hb = (3/16) * (35/pi)^(0.5) * ((x_coord^2)*(x_coord^2 - 3*y_coord^2) - (y_coord^2)*(3*x_coord^2 - y_coord^2))/(r^4);
    }
  
}



