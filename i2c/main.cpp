// #include <iostream.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <iostream>
// #include "i2c.h"
#include <linux/i2c-dev.h>
#include <i2c/smbus.h>

int main() {
    int check = 0;
    while(check==0){

    
    // Open bus on 0
    const char *device = "/dev/i2c-1";
    int deviceName = open(device, O_RDWR);
    // Specify STM address
    int addr = 0x40;
    if (ioctl(*device, I2C_SLAVE, addr) < 0) {
        std::cout<<"Address error"<<std::endl;
  }
  else{
    check = 1;
  }
    // Write a byte to slave
    unsigned char buffer[1];
    ssize_t size = sizeof(buffer);
    memset(buffer, 1, sizeof(buffer));
    int n = write(*device, buffer, size);
    if (n<0){
        std::cout<<"Write error"<<std::endl;
        exit(1);
    }
    close(*device);

}
    // Close bus
    
    return 0;
}