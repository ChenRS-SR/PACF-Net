# coding:UTF-8
import threading
import time
import serial
from serial import SerialException


# 串口配置
class SerialConfig:
    # 串口号
    portName = ''

    # 波特率
    baud = 9600


# 设备实例
class DeviceModel:
    # region 属性

    # 设备名称
    deviceName = "我的设备"

    # 设备modbus ID
    ADDR = 0x50

    # 设备数据字典
    deviceData = {}

    # 设备是否开启
    isOpen = False

    # 是否循环读取
    loop = False

    # 串口
    serialPort = None

    # 串口配置
    serialConfig = SerialConfig()

    # 临时数组
    TempBytes = []

    # 起始寄存器
    statReg = None

    # endregion

    # region   计算CRC
    auchCRCHi = [
        0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
        0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
        0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
        0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81,
        0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
        0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
        0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
        0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
        0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
        0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
        0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
        0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
        0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
        0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
        0x40]

    auchCRCLo = [
        0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06, 0x07, 0xC7, 0x05, 0xC5, 0xC4,
        0x04, 0xCC, 0x0C, 0x0D, 0xCD, 0x0F, 0xCF, 0xCE, 0x0E, 0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09,
        0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9, 0x1B, 0xDB, 0xDA, 0x1A, 0x1E, 0xDE, 0xDF, 0x1F, 0xDD,
        0x1D, 0x1C, 0xDC, 0x14, 0xD4, 0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
        0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3, 0xF2, 0x32, 0x36, 0xF6, 0xF7,
        0x37, 0xF5, 0x35, 0x34, 0xF4, 0x3C, 0xFC, 0xFD, 0x3D, 0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A,
        0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38, 0x28, 0xE8, 0xE9, 0x29, 0xEB, 0x2B, 0x2A, 0xEA, 0xEE,
        0x2E, 0x2F, 0xEF, 0x2D, 0xED, 0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
        0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60, 0x61, 0xA1, 0x63, 0xA3, 0xA2,
        0x62, 0x66, 0xA6, 0xA7, 0x67, 0xA5, 0x65, 0x64, 0xA4, 0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F,
        0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB, 0x69, 0xA9, 0xA8, 0x68, 0x78, 0xB8, 0xB9, 0x79, 0xBB,
        0x7B, 0x7A, 0xBA, 0xBE, 0x7E, 0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
        0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71, 0x70, 0xB0, 0x50, 0x90, 0x91,
        0x51, 0x93, 0x53, 0x52, 0x92, 0x96, 0x56, 0x57, 0x97, 0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C,
        0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E, 0x5A, 0x9A, 0x9B, 0x5B, 0x99, 0x59, 0x58, 0x98, 0x88,
        0x48, 0x49, 0x89, 0x4B, 0x8B, 0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
        0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42, 0x43, 0x83, 0x41, 0x81, 0x80,
        0x40]

    # endregion  计算CRC

    def __init__(self, deviceName, portName, baud, ADDR):
        print("初始化设备模型")
        # 设备名称（自定义）
        self.deviceName = deviceName
        # 串口号
        self.serialConfig.portName = portName
        # 串口波特率
        self.serialConfig.baud = baud
        # modbus 设备地址
        self.ADDR = ADDR

    # 获得CRC校验
    def get_crc(self, datas, dlen):
        tempH = 0xff  # 高 CRC 字节初始化 High CRC byte initialization
        tempL = 0xff  # 低 CRC 字节初始化 Low CRC byte initialization
        for i in range(0, dlen):
            tempIndex = (tempH ^ datas[i]) & 0xff
            tempH = (tempL ^ self.auchCRCHi[tempIndex]) & 0xff
            tempL = self.auchCRCLo[tempIndex]
        return (tempH << 8) | tempL
        pass

    # region 获取设备数据

    # 设置设备数据
    def set(self, key, value):
        # 将设备数据存到键值
        self.deviceData[key] = value

    # 获得设备数据
    def get(self, key):
        # 从键值中获取数据，没有则返回None
        if key in self.deviceData:
            return self.deviceData[key]
        else:
            return None

    # 删除设备数据
    def remove(self, key):
        # 删除设备键值
        del self.deviceData[key]

    # endregion

    # 打开设备
    def openDevice(self):
        # 先关闭端口
        self.closeDevice()
        try:
            self.serialPort = serial.Serial(self.serialConfig.portName, self.serialConfig.baud, timeout=0.5)
            self.isOpen = True
            print("{}已打开".format(self.serialConfig.portName))
            # 开启一个线程持续监听串口数据
            t = threading.Thread(target=self.readDataTh, args=("Data-Received-Thread", 10,))
            t.start()
            print("设备打开成功")
        except SerialException as e:
            print("打开" + self.serialConfig.portName + "失败")
            print(e)

    # 监听串口数据线程
    def readDataTh(self, threadName, delay):
        print("启动" + threadName)
        while True:
            # 如果串口打开了
            if self.isOpen:
                try:
                    tLen = self.serialPort.inWaiting()
                    if tLen > 0:
                        data = self.serialPort.read(tLen)
                        self.onDataReceived(data)
                except Exception as ex:
                    print(ex)
            else:
                time.sleep(0.1)
                print("串口未打开")
                break

    # 关闭设备
    def closeDevice(self):
        if self.serialPort is not None:
            self.serialPort.close()
            print("端口关闭了")
        self.isOpen = False
        print("设备关闭了")

    # region 数据解析

    # 串口数据处理
    def onDataReceived(self, data):
        temp_data = bytes.fromhex(data.hex())
        for val in temp_data:
            self.TempBytes.append(val)
            # 判断ID是否正确
            if self.TempBytes[0] != self.ADDR:
                del self.TempBytes[0]
                continue
            # 判断是否是03读取功能码
            if len(self.TempBytes) > 2:
                if not (self.TempBytes[1] == 0x03):
                    del self.TempBytes[0]
                    continue
                tLen = len(self.TempBytes)
                # 拿到一包完整协议数据
                if tLen == self.TempBytes[2] + 5:
                    # CRC校验
                    tempCrc = self.get_crc(self.TempBytes, tLen - 2)
                    if (tempCrc >> 8) == self.TempBytes[tLen - 2] and (tempCrc & 0xff) == self.TempBytes[tLen - 1]:
                        self.processData(self.TempBytes[2])
                    else:
                        del self.TempBytes[0]

    # 数据解析
    def processData(self, length):
        # 从读取指令中获得起始寄存器
        if self.statReg is not None:
            for i in range(int(length / 2)):
                # 寄存器数据
                value = self.TempBytes[2 * i + 3] << 8 | self.TempBytes[2 * i + 4]
                value = float(value)
                # 振动角度解析
                if 0x3D <= self.statReg <= 0x3F:
                    value = value / 32768.0* 180.0
                    self.set(str(self.statReg), value)
                    self.statReg += 1
                # 温度解析
                elif self.statReg == 0x40:
                    value = value / 100.0
                    self.set(str(self.statReg), value)
                    self.statReg += 1
                # 其他
                else:
                    self.set(str(self.statReg), value)
                    self.statReg += 1
            self.TempBytes.clear()

    # endregion

    # 发送串口数据
    def sendData(self, data):
        try:
            self.serialPort.write(data)
        except Exception as ex:
            print(ex)

    # 读取寄存器
    def readReg(self, regAddr, regCount):
        # 从指令中获取起始寄存器 （处理回传数据需要用到）
        self.statReg = regAddr
        # 封装读取指令并向串口发送数据
        self.sendData(self.get_readBytes(self.ADDR, regAddr, regCount))

    # 写入寄存器
    def writeReg(self, regAddr, sValue):
        # 解锁
        self.unlock()
        # 延迟100ms
        time.sleep(0.1)
        # 封装写入指令并向串口发送数据
        self.sendData(self.get_writeBytes(self.ADDR, regAddr, sValue))
        # 延迟100ms
        time.sleep(0.1)
        # 保存
        self.save()

    # 发送读取指令封装
    def get_readBytes(self, devid, regAddr, regCount):
        # 初始化
        tempBytes = [None] * 8
        # 设备modbus地址
        tempBytes[0] = devid
        # 读取功能码
        tempBytes[1] = 0x03
        # 寄存器高8位
        tempBytes[2] = regAddr >> 8
        # 寄存器低8位
        tempBytes[3] = regAddr & 0xff
        # 读取寄存器个数高8位
        tempBytes[4] = regCount >> 8
        # 读取寄存器个数低8位
        tempBytes[5] = regCount & 0xff
        # 获得CRC校验
        tempCrc = self.get_crc(tempBytes, len(tempBytes) - 2)
        # CRC校验高8位
        tempBytes[6] = tempCrc >> 8
        # CRC校验低8位
        tempBytes[7] = tempCrc & 0xff
        return tempBytes

    # 发送写入指令封装
    def get_writeBytes(self, devid, regAddr, sValue):
        # 初始化
        tempBytes = [None] * 8
        # 设备modbus地址
        tempBytes[0] = devid
        # 写入功能码
        tempBytes[1] = 0x06
        # 寄存器高8位
        tempBytes[2] = regAddr >> 8
        # 寄存器低8位
        tempBytes[3] = regAddr & 0xff
        # 寄存器值高8位
        tempBytes[4] = sValue >> 8
        # 寄存器值低8位
        tempBytes[5] = sValue & 0xff
        # 获得CRC校验
        tempCrc = self.get_crc(tempBytes, len(tempBytes) - 2)
        # CRC校验高8位
        tempBytes[6] = tempCrc >> 8
        # CRC校验低8位
        tempBytes[7] = tempCrc & 0xff
        return tempBytes

    # 开始循环读取
    def startLoopRead(self):
        # 循环读取控制
        self.loop = True
        # 开启读取线程
        t = threading.Thread(target=self.loopRead, args=())  # 开启一个线程接收数据
        t.start()

    # 循环读取线程
    def loopRead(self):
        print("循环读取开始")
        while self.loop:
            self.readReg(0x3A, 13)
            #self.readReg(0x34, 3)
            time.sleep(0.2)
        print("循环读取结束")

    # 关闭循环读取
    def stopLoopRead(self):
        self.loop = False

    # 解锁
    def unlock(self):
        cmd = self.get_writeBytes(self.ADDR, 0x69, 0xb588)
        self.sendData(cmd)

    # 保存
    def save(self):
        cmd = self.get_writeBytes(self.ADDR, 0x00, 0x0000)
        self.sendData(cmd)
