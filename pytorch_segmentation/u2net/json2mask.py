import json
import numpy as np
import cv2
from PIL import Image
import os


def parse_json_to_mask(json_path, output_mask_path,
                       background_color=(0, 0, 0),
                       label_colors=None):
    """
    解析JSON标注文件并生成对应的掩码图像。

    参数:
        json_path (str): 输入的JSON文件路径。
        output_mask_path (str): 输出的掩码图像路径。
        background_color (tuple): 背景颜色，默认为黑色 (B, G, R)。
        label_colors (dict): 标签到颜色的映射，例如 {"defect": (255, 0, 0)}。
    """
    if label_colors is None:
        # 默认颜色映射，红色用于“defect”
        label_colors = {
            "defect": (255, 0, 0),  # BGR格式
            # 你可以根据需要添加更多标签和颜色
            # "another_label": (0, 255, 0),
        }

    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析JSON文件 {json_path}: {e}")
        return
    except Exception as e:
        print(f"错误: 读取JSON文件 {json_path} 时出错: {e}")
        return

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")

    if image_width is None or image_height is None:
        print(f"错误: JSON文件 {json_path} 中缺少 imageWidth 或 imageHeight 信息。")
        return

    # 创建一个全背景色的图像
    # 使用OpenCV的BGR格式
    mask = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    shapes = data.get("shapes", [])
    for shape in shapes:
        label = shape.get("label")
        points = shape.get("points", [])

        if label not in label_colors:
            print(f"警告: 标签 '{label}' 未在颜色映射中定义，跳过此标签。")
            continue

        color = label_colors[label]

        # 转换点为numpy数组并转换为int32
        try:
            polygon = np.array(points, dtype=np.int32)
            if polygon.ndim != 2 or polygon.shape[1] != 2:
                print(f"警告: JSON文件 {json_path} 中的标签 '{label}' 的点格式不正确，跳过此标签。")
                continue
            polygon = polygon.reshape((-1, 1, 2))  # 适应 cv2.fillPoly 的输入格式
        except Exception as e:
            print(f"警告: 处理 JSON文件 {json_path} 中的标签 '{label}' 时出错: {e}")
            continue

        # 填充多边形
        cv2.fillPoly(mask, [polygon], color)

    # 将 BGR 转换为 RGB 以便使用 Pillow 保存
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # 使用 Pillow 保存图像
    try:
        mask_image = Image.fromarray(mask_rgb)
        mask_image.save(output_mask_path)
        print(f"成功: 掩码图像已保存到: {output_mask_path}")
    except Exception as e:
        print(f"错误: 保存掩码图像到 {output_mask_path} 时出错: {e}")


def process_all_json(input_dir, output_dir, mask_extension,
                     background_color=(0, 0, 0),
                     label_colors=None):
    """
    处理输入目录下的所有 JSON 文件，生成对应的掩码图像。

    参数:
        input_dir (str): 输入的JSON文件目录。
        output_dir (str): 输出的掩码图像目录。
        mask_extension (str): 掩码图像的扩展名（例如 '.png'）。
        background_color (tuple): 背景颜色，默认为黑色 (B, G, R)。
        label_colors (dict): 标签到颜色的映射，例如 {"defect": (255, 0, 0)}。
    """
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"信息: 输出目录已创建: {output_dir}")
        except Exception as e:
            print(f"错误: 无法创建输出目录 {output_dir}: {e}")
            return

    # 确保 mask_extension 以点开头
    if not mask_extension.startswith('.'):
        mask_extension = '.' + mask_extension

    # 遍历输入目录中的所有 JSON 文件
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]

    if not json_files:
        print(f"警告: 输入目录 {input_dir} 中没有找到 JSON 文件。")
        return

    print(f"信息: 在 {input_dir} 中找到 {len(json_files)} 个 JSON 文件。")

    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        base_name = os.path.splitext(json_file)[0]
        mask_file = base_name + mask_extension
        output_mask_path = os.path.join(output_dir, mask_file)

        parse_json_to_mask(
            json_path=json_path,
            output_mask_path=output_mask_path,
            background_color=background_color,
            label_colors=label_colors
        )


if __name__ == "__main__":
    # =========================
    # 用户需要修改的部分
    # =========================

    # 定义输入JSON文件目录路径
    input_dir = "D:\800data\整合\.图像整合"  # 替换为你的JSON文件目录路径

    # 定义输出掩码图像目录路径
    output_dir = input_dir + "\masks"  # 替换为你想保存掩码图像的目录路径

    # 定义掩码图像的扩展名
    mask_extension = ".png"  # 例如 ".png"、".jpg"

    # 定义背景颜色和标签颜色（B, G, R）
    background_color = (0, 0, 0)  # 黑色

    # 标签到颜色的映射（B, G, R）
    label_colors = {
        "defect": (0, 0, 255),  # BGR
        # 添加更多标签及其颜色，例如：
        # "scratch": (0, 255, 0),  # 绿色
        # "dent": (0, 0, 255),     # 蓝色
    }

    # =========================
    # 脚本执行部分
    # =========================

    process_all_json(
        input_dir=input_dir,
        output_dir=output_dir,
        mask_extension=mask_extension,
        background_color=background_color,
        label_colors=label_colors
    )
