"""
批量图片标定工具
功能：
- 选择图片文件夹（如 Computer_Camera）
- 指定起始图片编号
- 逐张标定图片中的特征点
- 右方向键保存当前点并打开下一张
- 左方向键保存当前点并打开前一张
- 所有标定点保存到 CSV 文件（位于上级目录）
"""

import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime


class ImageMarkerBatch:
    """批量图片标定工具"""
    
    def __init__(self, image_dir, start_num=1):
        """
        初始化标定工具
        
        Args:
            image_dir: 图片文件夹路径 (如 Computer_Camera)
            start_num: 起始图片编号
        """
        self.image_dir = Path(image_dir)
        self.parent_dir = self.image_dir.parent  # 上级目录 (images)
        self.camera_name = self.image_dir.name  # 相机名称
        
        # CSV 文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"{self.camera_name}_calibration_{timestamp}.csv"
        self.csv_path = self.parent_dir / self.csv_filename
        
        # 当前图片编号
        self.current_num = start_num
        
        # 标记信息
        self.marked_point = None
        self.current_image = None
        self.display_image = None
        self.window_name = f"图片标定工具 - {self.camera_name}"
        
        # 标定数据 (用于保存)
        self.calibration_data = []
        self.load_existing_csv()
        
    def load_existing_csv(self):
        """加载已有的 CSV 文件"""
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                self.calibration_data = df.to_dict('records')
                print(f"✓ 加载已有标定数据: {self.csv_filename}")
                print(f"  共 {len(self.calibration_data)} 条记录")
            except Exception as e:
                print(f"⚠️  加载 CSV 失败: {e}")
                self.calibration_data = []
        else:
            print(f"✓ 创建新的标定文件: {self.csv_filename}")
            self.calibration_data = []
    
    def load_image(self, img_num):
        """
        加载图片
        
        Args:
            img_num: 图片编号
            
        Returns:
            成功返回 True
        """
        image_name = f"image-{img_num}.jpg"
        image_path = self.image_dir / image_name
        
        if not image_path.exists():
            print(f"❌ 图片不存在: {image_path}")
            return False
        
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"❌ 无法加载图片: {image_path}")
            return False
        
        self.current_num = img_num
        self.display_image = self.current_image.copy()
        
        # 尝试加载已有的标定点
        self.marked_point = self.get_existing_point(img_num)
        
        print(f"\n✓ 加载图片: {image_name} [{img_num}]")
        print(f"  尺寸: {self.display_image.shape[1]}x{self.display_image.shape[0]}")
        
        if self.marked_point:
            print(f"  已有标定点: {self.marked_point}")
        
        return True
    
    def get_existing_point(self, img_num):
        """获取已有的标定点"""
        image_name = f"image-{img_num}.jpg"
        for record in self.calibration_data:
            if record.get('image') == image_name:
                u = int(record.get('u', 0))
                v = int(record.get('v', 0))
                return (u, v)
        return None
    
    def draw_marked_point(self):
        """绘制标记点"""
        self.display_image = self.current_image.copy()
        
        if self.marked_point:
            u, v = self.marked_point
            # 绘制圆点 (蓝色)
            cv2.circle(self.display_image, (u, v), 8, (255, 0, 0), -1)
            # 绘制十字线
            cv2.line(self.display_image, (u - 20, v), (u + 20, v), (255, 0, 0), 2)
            cv2.line(self.display_image, (u, v - 20), (u, v + 20), (255, 0, 0), 2)
            # 显示坐标
            cv2.putText(self.display_image, f"({u}, {v})", 
                       (u + 30, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 0, 0), 2)
        
        # 显示操作提示
        self.draw_instructions()
        cv2.imshow(self.window_name, self.display_image)
    
    def draw_instructions(self):
        """在图片上显示操作提示"""
        h, w = self.display_image.shape[:2]
        
        # 背景框
        cv2.rectangle(self.display_image, (10, 10), (400, 150), (200, 200, 200), -1)
        cv2.rectangle(self.display_image, (10, 10), (400, 150), (0, 0, 0), 2)
        
        # 文字提示
        y_offset = 35
        cv2.putText(self.display_image, "Click: Mark point", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(self.display_image, "D: Save & Next", 
                   (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(self.display_image, "A: Save & Prev", 
                   (20, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(self.display_image, "Q: Quit", 
                   (20, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.marked_point = (x, y)
            print(f"📍 标记点: ({x}, {y})")
            self.draw_marked_point()
    
    def save_current_point(self):
        """保存当前标记点到 CSV"""
        if self.marked_point is None:
            print("⚠️  未标记任何点，跳过保存")
            return False
        
        image_name = f"image-{self.current_num}.jpg"
        u, v = self.marked_point
        
        # 检查是否已存在该图片的记录
        existing_index = -1
        for i, record in enumerate(self.calibration_data):
            if record.get('image') == image_name:
                existing_index = i
                break
        
        new_record = {
            'image': image_name,
            'u': u,
            'v': v,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if existing_index >= 0:
            # 更新已有记录
            self.calibration_data[existing_index] = new_record
            print(f"✓ 更新标定点: {image_name} -> ({u}, {v})")
        else:
            # 添加新记录
            self.calibration_data.append(new_record)
            print(f"✓ 保存标定点: {image_name} -> ({u}, {v})")
        
        # 写入 CSV
        self.write_csv()
        return True
    
    def write_csv(self):
        """写入 CSV 文件"""
        try:
            df = pd.DataFrame(self.calibration_data)
            df.to_csv(self.csv_path, index=False, encoding='utf-8')
            print(f"  💾 已保存到: {self.csv_path}")
        except Exception as e:
            print(f"❌ 保存 CSV 失败: {e}")
    
    def next_image(self):
        """打开下一张图片"""
        if not self.save_current_point():
            return False
        
        if not self.load_image(self.current_num + 1):
            print("⚠️  已是最后一张图片")
            self.current_num -= 1  # 恢复编号
            return False
        
        self.draw_marked_point()
        return True
    
    def prev_image(self):
        """打开前一张图片"""
        if not self.save_current_point():
            return False
        
        if self.current_num <= 1:
            print("⚠️  已是第一张图片")
            self.current_num += 1  # 恢复编号
            return False
        
        if not self.load_image(self.current_num - 1):
            print("⚠️  找不到前一张图片")
            self.current_num += 1  # 恢复编号
            return False
        
        self.draw_marked_point()
        return True
    
    def run(self):
        """运行标定工具"""
        print(f"\n{'#'*70}")
        print(f"# 批量图片标定工具")
        print(f"{'#'*70}\n")
        
        # 加载第一张图片
        if not self.load_image(self.current_num):
            return False
        
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 绘制初始显示
        self.draw_marked_point()
        
        print("\n" + "="*70)
        print("📌 操作说明:")
        print("  🖱️  左击图片: 标记标定点")
        print("  D 键: 保存当前点 → 打开下一张")
        print("  A 键: 保存当前点 → 打开前一张")
        print("  Q 键: 保存所有数据并退出")
        print("="*70 + "\n")
        
        # 等待用户操作
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('d') or key == ord('D'):  # D 键：下一张
                self.next_image()
            elif key == ord('a') or key == ord('A'):  # A 键：上一张
                self.prev_image()
            elif key == ord('q') or key == ord('Q'):
                # 保存最后一张的标记点
                if self.marked_point:
                    self.save_current_point()
                print("\n✓ 标定完成！")
                break
        
        # 关闭窗口
        cv2.destroyAllWindows()
        
        print(f"\n{'='*70}")
        print(f"📊 标定统计:")
        print(f"  总标定点数: {len(self.calibration_data)}")
        print(f"  CSV 保存路径: {self.csv_path}")
        print(f"{'='*70}\n")
        
        return True


def main():
    """主函数"""
    print("\n" + "="*70)
    print("批量图片标定工具")
    print("="*70)
    
    # 获取图片文件夹
    image_dir = input("\n请输入图片文件夹路径: ").strip()
    if not image_dir:
        print("❌ 文件夹路径不能为空")
        return
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"❌ 文件夹不存在: {image_dir}")
        return
    
    # 获取起始编号
    start_num_str = input("请输入起始图片编号: ").strip()
    try:
        start_num = int(start_num_str)
    except ValueError:
        print("❌ 编号必须是数字")
        return
    
    # 创建标定工具并运行
    marker = ImageMarkerBatch(image_dir, start_num)
    marker.run()


if __name__ == "__main__":
    main()
