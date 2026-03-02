"""
交互式图片裁剪调试工具
用于手动标记裁剪中心点并生成 debug 图片
支持多种相机：Computer_Camera 和 Fotric_Camera (热像)
流程：
1. 选择相机类型 (1=Computer_Camera, 2=Fotric_Camera)
2. 输入源数据文件夹路径
3. 输入图片编号
4. 显示原始图片 (Computer_Camera 的 RGB 或 Fotric 的伪彩图)
5. 鼠标点击标记裁剪中心点
6. 生成并保存 debug 可视化图片
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


class InteractiveCropDebugger:
    """交互式裁剪调试工具"""
    
    def __init__(self, source_dir, output_dir, camera_type=1, fotric_data_source=1):
        """
        初始化调试工具
        
        Args:
            source_dir: 源数据目录 (包含 images/)
            output_dir: 输出目录 (用于保存 debug 图片)
            camera_type: 相机类型 (1=Computer_Camera, 2=Fotric_Camera)
            fotric_data_source: Fotric 数据来源 (1=温度矩阵NPZ, 2=伪彩色图片JPG)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.camera_type = camera_type
        self.fotric_data_source = fotric_data_source
        
        if camera_type == 1:
            # Computer_Camera 配置
            self.camera_name = "Computer_Camera"
            self.image_dir = self.source_dir / "images" / "Computer_Camera"
            self.crop_size = 224
            self.output_debug_dir = self.output_dir / "Manual_Debug_Computer"
        else:
            # Fotric_Camera 配置
            self.camera_name = "Fotric_Camera"
            if fotric_data_source == 1:
                self.image_dir = self.source_dir / "images" / "Fotric_Data"
                self.data_source_name = "温度矩阵 (NPZ)"
            else:
                self.image_dir = self.source_dir / "images" / "Fotric_Camera"
                self.data_source_name = "伪彩色图片 (JPG)"
            
            self.crop_size = 112
            self.output_debug_dir = self.output_dir / "Manual_Debug_Fotric"
        
        self.crop_half = self.crop_size // 2
        
        # 标记信息
        self.marked_point = None
        self.original_image = None
        self.display_image = None
        self.window_name = f"裁剪调试工具 ({self.camera_name}) - 点击标记裁剪中心，右上角点击退出"
        
    def create_output_dir(self):
        """创建输出目录"""
        self.output_debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建输出目录: {self.output_debug_dir}")
    
    def load_image(self, img_num):
        """
        加载图片
        
        Args:
            img_num: 图片编号
            
        Returns:
            成功返回 True
        """
        if self.camera_type == 1:
            # Computer_Camera - 加载 JPG 图片
            image_name = f"image-{img_num}.jpg"
            image_path = self.image_dir / image_name
            
            if not image_path.exists():
                print(f"❌ 图片不存在: {image_path}")
                return False
            
            self.original_image = cv2.imread(str(image_path))
            if self.original_image is None:
                print(f"❌ 无法加载图片: {image_path}")
                return False
        else:
            # Fotric_Camera
            if self.fotric_data_source == 1:
                # 方式 1：从 NPZ 文件加载温度数据
                npz_name = f"thermal_data-{img_num}.npz"
                npz_path = self.image_dir / npz_name
                
                if not npz_path.exists():
                    print(f"❌ NPZ 文件不存在: {npz_path}")
                    return False
                
                try:
                    data = np.load(npz_path)
                    temps = data['thermal_data']
                    
                    # 温度数据归一化为灰度图
                    temps_normalized = (temps - 20) / (250 - 20)
                    temps_normalized = np.clip(temps_normalized, 0.0, 1.0)
                    temps_vis = (temps_normalized * 255).astype(np.uint8)
                    
                    # 转换为 BGR 格式便于显示
                    self.original_image = cv2.cvtColor(temps_vis, cv2.COLOR_GRAY2BGR)
                    
                except Exception as e:
                    print(f"❌ 读取 NPZ 文件失败: {e}")
                    return False
            else:
                # 方式 2：直接加载伪彩色图片
                image_name = f"image-{img_num}.jpg"
                image_path = self.image_dir / image_name
                
                if not image_path.exists():
                    print(f"❌ 伪彩色图片不存在: {image_path}")
                    return False
                
                self.original_image = cv2.imread(str(image_path))
                if self.original_image is None:
                    print(f"❌ 无法加载伪彩色图片: {image_path}")
                    return False
        
        self.display_image = self.original_image.copy()
        
        if self.camera_type == 1:
            print(f"✓ 成功加载 {self.camera_name} 图片: image-{img_num}")
        else:
            print(f"✓ 成功加载 {self.camera_name} 图片: image-{img_num} ({self.data_source_name})")
        
        print(f"  图片尺寸: {self.display_image.shape[1]}x{self.display_image.shape[0]}")
        return True
    
    def draw_exit_button(self):
        """在右上角绘制退出按钮"""
        h, w = self.display_image.shape[:2]
        button_w, button_h = 80, 40
        x = w - button_w - 10
        y = 10
        
        # 绘制按钮背景
        cv2.rectangle(self.display_image, (x, y), (x + button_w, y + button_h), (0, 0, 255), -1)
        # 绘制按钮边框
        cv2.rectangle(self.display_image, (x, y), (x + button_w, y + button_h), (0, 0, 200), 2)
        # 绘制文字
        cv2.putText(self.display_image, "EXIT", (x + 15, y + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存按钮区域
        self.button_area = (x, y, x + button_w, y + button_h)
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击了退出按钮
            if hasattr(self, 'button_area'):
                bx1, by1, bx2, by2 = self.button_area
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    print("\n✓ 点击退出按钮")
                    return
            
            # 记录标记点
            self.marked_point = (x, y)
            print(f"📍 标记中心点: ({x}, {y})")
            
            # 重新绘制
            self.redraw()
    
    def redraw(self):
        """重新绘制图片"""
        self.display_image = self.original_image.copy()
        
        # 绘制已标记的点
        if self.marked_point:
            center_u, center_v = self.marked_point
            left = max(0, center_u - self.crop_half)
            right = min(self.display_image.shape[1], center_u + self.crop_half)
            top = max(0, center_v - self.crop_half)
            bottom = min(self.display_image.shape[0], center_v + self.crop_half)
            
            # 绘制裁剪区域矩形 (红色线条)
            cv2.rectangle(self.display_image, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # 绘制中心点 (红色实心圆)
            cv2.circle(self.display_image, (center_u, center_v), 8, (0, 0, 255), -1)
            # 绘制十字线
            cv2.line(self.display_image, (center_u - 15, center_v), (center_u + 15, center_v), (0, 0, 255), 2)
            cv2.line(self.display_image, (center_u, center_v - 15), (center_u, center_v + 15), (0, 0, 255), 2)
            
            # 添加文字标注
            cv2.putText(self.display_image, f"Center: ({center_u}, {center_v})", 
                       (center_u + 20, center_v - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 1)
            cv2.putText(self.display_image, f"Size: {self.crop_size}x{self.crop_size}", 
                       (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 1)
        
        # 绘制退出按钮
        self.draw_exit_button()
        cv2.imshow(self.window_name, self.display_image)
    
    def save_debug_image(self, img_num):
        """
        保存 debug 图片
        
        Args:
            img_num: 图片编号
            
        Returns:
            成功返回 True
        """
        if self.marked_point is None:
            print("❌ 没有标记点，无法保存")
            return False
        
        try:
            debug_image = self.original_image.copy()
            center_u, center_v = self.marked_point
            left = max(0, center_u - self.crop_half)
            right = min(debug_image.shape[1], center_u + self.crop_half)
            top = max(0, center_v - self.crop_half)
            bottom = min(debug_image.shape[0], center_v + self.crop_half)
            
            # 绘制裁剪区域矩形
            cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # 绘制中心点
            cv2.circle(debug_image, (center_u, center_v), 8, (0, 0, 255), -1)
            cv2.line(debug_image, (center_u - 15, center_v), (center_u + 15, center_v), (0, 0, 255), 2)
            cv2.line(debug_image, (center_u, center_v - 15), (center_u, center_v + 15), (0, 0, 255), 2)
            
            # 添加文字标注
            cv2.putText(debug_image, f"Center: ({center_u}, {center_v})", 
                       (center_u + 20, center_v - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 1)
            cv2.putText(debug_image, f"Size: {self.crop_size}x{self.crop_size}", 
                       (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 1)
            
            # 转换 BGR 到 RGB (PIL 需要 RGB)
            debug_image_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            
            # 使用 PIL 保存为高质量 PNG (300 DPI)
            image_name = f"image-{img_num}_debug.png"
            output_path = self.output_debug_dir / image_name
            pil_image = Image.fromarray(debug_image_rgb)
            pil_image.save(str(output_path), dpi=(300, 300), quality=95)
            
            print(f"✓ 保存高清 debug 图片: {output_path}")
            print(f"  DPI: 300, 格式: PNG")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
    
    def run(self, img_num):
        """
        运行交互式调试工具
        
        Args:
            img_num: 图片编号
        """
        print(f"\n{'#'*60}")
        print(f"# 交互式裁剪调试工具")
        print(f"{'#'*60}\n")
        
        # 创建输出目录
        self.create_output_dir()
        
        # 加载图片
        if not self.load_image(img_num):
            return False
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 绘制初始显示
        self.draw_exit_button()
        cv2.imshow(self.window_name, self.display_image)
        
        print("\n" + "="*60)
        print("📌 使用说明:")
        print("  - 左击图片标记裁剪中心点")
        print("  - 右上角点击 EXIT 按钮或按 Q 键完成")
        print("="*60 + "\n")
        
        # 等待用户操作
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\n✓ 按 Q 键完成")
                break
        
        # 关闭窗口
        cv2.destroyAllWindows()
        
        # 保存 debug 图片
        if self.marked_point:
            self.save_debug_image(img_num)
            print(f"\n✓ 完成！")
            print(f"  标记点: {self.marked_point}")
            print(f"  输出目录: {self.output_debug_dir}")
            return True
        else:
            print("\n⊘ 未标记任何点")
            return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("交互式图片裁剪调试工具")
    print("="*60)
    
    # 选择相机类型
    print("\n📷 选择相机类型:")
    print("  1 - Computer_Camera (224×224)")
    print("  2 - Fotric_Camera (112×112，热像)")
    
    camera_choice = input("\n请选择 (1 或 2): ").strip()
    if camera_choice not in ['1', '2']:
        print("❌ 选择无效，请输入 1 或 2")
        return
    
    camera_type = int(camera_choice)
    fotric_data_source = 1  # 默认值
    
    # 如果选择 Fotric_Camera，再选择数据来源
    if camera_type == 2:
        print("\n🌡️  选择 Fotric 数据来源:")
        print("  1 - 温度矩阵 (NPZ 文件，灰度图)")
        print("  2 - 伪彩色图片 (JPG 文件，彩色图)")
        
        fotric_choice = input("\n请选择 (1 或 2): ").strip()
        if fotric_choice not in ['1', '2']:
            print("❌ 选择无效，请输入 1 或 2")
            return
        fotric_data_source = int(fotric_choice)
    
    # 获取输入
    source_dir = input("\n请输入源数据目录 (包含 images/): ").strip()
    if not source_dir:
        source_dir = r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01\task_20260122_202652"
        print(f"使用默认路径: {source_dir}")
    
    output_dir = input("请输入输出目录: ").strip()
    if not output_dir:
        output_dir = r"D:\College\Python_project\4Project\data\FDMdata\202601FDM_Series01_Crop\task_20260122_202652"
        print(f"使用默认路径: {output_dir}")
    
    img_num = input("请输入图片编号 (如 996): ").strip()
    if not img_num:
        print("❌ 图片编号不能为空")
        return
    
    try:
        img_num = int(img_num)
    except ValueError:
        print("❌ 图片编号必须是数字")
        return
    
    # 创建调试工具并运行
    debugger = InteractiveCropDebugger(source_dir, output_dir, camera_type, fotric_data_source)
    debugger.run(img_num)


if __name__ == "__main__":
    main()
