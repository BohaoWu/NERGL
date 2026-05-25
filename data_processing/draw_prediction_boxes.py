import os
import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """读取图像文件"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"图像无法读取，可能是格式错误: {image_path}")
    return image

def load_predictions(npz_path: str):
    """读取VinVL提取的预测框与标签"""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"预测特征文件不存在: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    boxes = data.get('boxes')
    if boxes is None:
        raise KeyError("'boxes' 不存在于 npz 文件中")

    names = data.get('names')
    if names is None:
        names = [f"obj_{i}" for i in range(len(boxes))]
    return boxes, names

def draw_boxes_on_image(image: np.ndarray, boxes: np.ndarray, names: list[str]):
    """在图像上绘制预测框和标签"""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = str(names[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image, label, (x1, max(y1 - 5, 0)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 255, 0), thickness=1)
    return image

def save_image(image: np.ndarray, output_path: str) -> None:
    """保存处理后的图像到指定路径"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"[INFO] 图像保存成功: {output_path}")

def main():
    image_path = 'images/NDL-107-00-045.jpg'
    npz_path = 'features/NDL-107-00-045.npz'
    output_path = 'outputs/NDL-107-00-045.jpg'

    image = load_image(image_path)
    boxes, names = load_predictions(npz_path)
    image_with_boxes = draw_boxes_on_image(image, boxes, names)
    save_image(image_with_boxes, output_path)

if __name__ == '__main__':
    main()