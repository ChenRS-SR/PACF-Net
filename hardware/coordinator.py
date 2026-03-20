#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ 完整的M114坐标提取解决方案
通过实时监听OctoPrint的serial.log来获取M114响应

工作原理：
1. 发送M114命令通过REST API
2. 立即开始监听serial.log文件的新增内容
3. 检测到M114响应后提取坐标
4. 返回X,Y,Z坐标值
"""

import os
import re
import time
import requests
import json
from threading import Thread, Lock
from datetime import datetime
from configs.collector_config import OCTOPRINT_CONFIG

OCTOPRINT_URL = OCTOPRINT_CONFIG['url']
API_KEY = OCTOPRINT_CONFIG['api_key']

class M114Coordinator:
    """通过serial.log获取打印机坐标"""
    
    def __init__(self):
        self.log_file = os.path.expanduser("~/AppData/Roaming/OctoPrint/logs/serial.log")
        self.coordinates = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.lock = Lock()
        self.last_position = 0  # 日志文件上次读到的位置
        
        # 初始化日志位置
        if os.path.exists(self.log_file):
            self.last_position = os.path.getsize(self.log_file)
            print("[M114] Serial log found at: {}".format(self.log_file))
            print("[M114] Log file size: {} bytes".format(self.last_position))
        else:
            print("[M114ERROR] Serial log file NOT found!")
            print("[M114ERROR] Expected path: {}".format(self.log_file))
            print("[M114WARNING] Coordinates may not be available")
    
    def send_m114(self):
        """发送M114命令到打印机"""
        headers = {"X-Api-Key": API_KEY}
        try:
            resp = requests.post(
                f"{OCTOPRINT_URL}/api/printer/command",
                headers=headers,
                json={"command": "M114"},
                timeout=5
            )
            print("[M114] Send success (status: {})".format(resp.status_code))
            return True
        except Exception as e:
            print("[M114] Send failed: {}".format(e))
            return False
    
    def wait_for_m114_response(self, timeout=5):
        """
        发送M114后，等待并解析响应
        """
        # 检查日志文件是否存在
        if not os.path.exists(self.log_file):
            print("[M114ERROR] Serial.log file not found at: {}".format(self.log_file))
            return None
        
        # 静默等待M114响应，只在出错时输出
        
        # 发送命令
        if not self.send_m114():
            return None
        
        # 监听日志
        start_time = time.time()
        m114_found = False
        
        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(self.log_file)
                
                # 检查文件是否有新增内容
                if current_size > self.last_position:
                    with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(self.last_position)
                        new_content = f.read()
                        self.last_position = f.tell()
                    
                    # 在新内容中查找M114和响应
                    lines = new_content.split('\n')
                    for i, line in enumerate(lines):
                        # 寻找M114发送命令
                        if 'Send: M114' in line:
                            m114_found = True
                            pass  # 找到M114命令，静默处理
                        
                        # 如果M114已经发送过，就在后续行中寻找坐标响应
                        if m114_found and 'X:' in line and 'Y:' in line and 'Z:' in line:
                            # 确保这是Recv行（接收）
                            if 'Recv:' in line:
                                pass  # 收到响应，静默处理
                                
                                # 提取坐标
                                coords = self.extract_coordinates(line)
                                if coords:
                                    # 坐标获取成功，由调用者决定是否输出
                                    with self.lock:
                                        self.coordinates = coords
                                    return coords
                
                time.sleep(0.05)
                
            except Exception as e:
                print("[ERROR] {}".format(e))
                time.sleep(0.1)
        
        print("[TIMEOUT] No M114 response in {} seconds".format(timeout))
        return None
    
    def extract_coordinates(self, line):
        """
        从serial.log行提取X,Y,Z坐标
        格式: "2026-01-22 19:43:04,921 - Recv: X:117.49 Y:107.44 Z:7.20 E:566.01 Count X:9400 Y:9301 Z:1025"
        """
        try:
            # 使用正则表达式提取坐标
            match = re.search(r'X:([\d.]+)\s+Y:([\d.]+)\s+Z:([\d.]+)', line)
            if match:
                x, y, z = match.groups()
                return {
                    "X": float(x),
                    "Y": float(y),
                    "Z": float(z)
                }
        except Exception as e:
            print("[PARSE ERROR] {}".format(e))
        
        return None
    
    def get_current_coordinates(self):
        """获取最后一次读取的坐标"""
        with self.lock:
            return dict(self.coordinates)

def continuous_coordinate_monitoring():
    """
    持续监控坐标（每1.5秒获取一次）
    """
    print("=" * 70)
    print("Continuous coordinate monitoring (every 1.5 sec)")
    print("=" * 70)
    
    coord = M114Coordinator()
    
    try:
        for i in range(20):  # 采集20次
            print("\n[Collection #{}]".format(i+1))
            coords = coord.wait_for_m114_response(timeout=3)
            
            if coords:
                print("X: {:.2f} mm".format(coords['X']))
                print("Y: {:.2f} mm".format(coords['Y']))
                print("Z: {:.2f} mm".format(coords['Z']))
            else:
                print("Failed to get coordinates")
            
            # 等待一下再采集
            if i < 19:
                time.sleep(1.5)
    
    except KeyboardInterrupt:
        print("\n\nStopped")

def single_coordinate_test():
    """
    单次坐标获取测试
    """
    print("=" * 70)
    print("Single M114 coordinate test")
    print("=" * 70)
    
    coord = M114Coordinator()
    coords = coord.wait_for_m114_response(timeout=5)
    
    if coords:
        print("\nSuccess!")
        print("X = {}".format(coords['X']))
        print("Y = {}".format(coords['Y']))
        print("Z = {}".format(coords['Z']))
    else:
        print("\nFailed")

if __name__ == "__main__":
    print("\nOctoPrint M114 Coordinate Acquisition System\n")
    
    # 测试单次
    single_coordinate_test()
    
    # 如果需要连续监控，取消注释下一行
    # continuous_coordinate_monitoring()
