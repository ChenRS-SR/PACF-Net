"""
摄像头检测设备脚本
运行此脚本查看系统中可用的摄像头设备号
"""

import cv2
import numpy as np

def check_cameras(max_id=10):
    """检测可用的摄像头设备"""
    print("=" * 60)
    print("摄像头设备检测")
    print("=" * 60)
    print()
    
    available_cameras = []
    
    for i in range(max_id):
        print(f"检查设备 {i}...", end=" ")
        
        # 尝试DirectShow后端
        cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # 获取基本信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 尝试读取一帧
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # 检测亮度
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                
                print(f"✓ 可用")
                print(f"    分辨率: {width}x{height}")
                print(f"    FPS: {fps:.1f}")
                print(f"    亮度: {brightness:.1f}")
                
                # 判断是否是虚拟摄像头
                if brightness < 15:
                    print(f"    ⚠️  警告: 画面太暗，可能是虚拟摄像头")
                elif brightness < 30 and np.std(gray) < 10:
                    print(f"    ⚠️  警告: 画面过于均匀，可能是虚拟摄像头")
                else:
                    print(f"    ✓ 看起来是真实摄像头")
                
                available_cameras.append({
                    'id': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'brightness': brightness
                })
            else:
                print(f"✗ 无法读取画面")
            
            cap.release()
        else:
            print(f"✗ 无法打开")
    
    print()
    print("=" * 60)
    print("检测总结")
    print("=" * 60)
    
    if available_cameras:
        print(f"\n发现 {len(available_cameras)} 个可用摄像头:\n")
        for cam in available_cameras:
            print(f"  设备 {cam['id']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}FPS")
        
        print()
        print("建议配置:")
        print("-" * 40)
        if len(available_cameras) >= 2:
            print(f'  旁轴相机 (computer): device_id = {available_cameras[1]["id"]}')
            print(f'  随轴相机 (ids): device_ids = [{available_cameras[0]["id"]}, ...]')
        else:
            print(f'  随轴相机 (ids): device_ids = [{available_cameras[0]["id"]}]')
    else:
        print("\n✗ 未发现可用摄像头")
        print("\n请检查:")
        print("1. 摄像头是否正确连接")
        print("2. 摄像头驱动是否安装")
        print("3. 摄像头是否被其他程序占用")
    
    print()
    print("=" * 60)
    print("配置文件位置: configs/collector_config.py")
    print("=" * 60)

if __name__ == "__main__":
    check_cameras()
