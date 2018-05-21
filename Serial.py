

##CLASS TO SEND DATA OVER SERIAL USB##
##USED TO COMMUNICATE WITH DUE IN PROJECT#
import serial


class Serial:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate)
        self.ser.parity = serial.PARITY_NONE
        self.ser.stopbits = serial.STOPBITS_ONE
        print('Connected to:',self.ser.port)

    #sends x and y coordinate to the reciever
    def sendData(self, x, y):
        self.ser.write(bytes(b'V'))
        testX = x // 2
        testY = y // 2
        print(x, y)
        #can only handle 1 byte
        if (testX <= 255) & (testX >= 0) & (testY <= 255) & (testY >= 0):
            self.ser.write(bytes([testX]))
            self.ser.write(bytes([testY]))
            print("Success")


