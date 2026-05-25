import os
import cv2
import numpy as np
import kanjize

import re
import ast

from PIL import Image, ImageDraw, ImageFont

import xml.etree.ElementTree as ET

class ResDrawer:
    def __init__(self):
        self.resource_path = "/root/GMNER/ukiyoe_picture/"
        self.npz_path = "/root/GMNER/Ukiyoe1000_VinVL/"
        self.anaotation_path = "/root/GMNER/Ukiyoe1000/xml/"
        self.save_path = "/root/GMNER/figure/res/"

    def draw_boxes_on_image(self, image_path, boxes, color=(0, 255, 0), thickness=2, save_path=None, cropped=False):
        """
        在图像上绘制多个 bounding box 和日文标签（使用 PIL 支持中文/日文字体）。
        
        Args:
            image_path (str): 图像路径
            boxes (List or ndarray): [N, 4]，每行为 (x1, y1, x2, y2)
            entities (List[str]): 对应的标签文字
            color (tuple): 框的颜色 (B, G, R)
            thickness (int): 线条粗细
            save_path (str): 若不为 None，则保存到该路径
        """
        # 使用支持日文的字体（确保路径正确）
        font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
        font = ImageFont.truetype(font_path, 36)

        # 读取图像（OpenCV读取 → RGB → PIL）
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)

        count = 0
        # 框和文字
        for box in boxes:
            count += 1
            x1, y1, x2, y2 = map(int, box)

            # ✅ 用 PIL 画框
            for i in range(thickness):  # thicknessを表現
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        # 转回BGR保存
        image_result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        if save_path:
            cv2.imwrite(save_path, image_result)
    
    def draw_boxes_and_label_on_image(self, image_path, boxes, entities=[], color=(0, 255, 0), thickness=6, save_path=None, cropped=False):
        """
        在图像上绘制多个 bounding box 和日文标签（使用 PIL 支持中文/日文字体）。
        
        Args:
            image_path (str): 图像路径
            boxes (List or ndarray): [N, 4]，每行为 (x1, y1, x2, y2)
            entities (List[str]): 对应的标签文字
            color (tuple): 框的颜色 (B, G, R)
            thickness (int): 线条粗细
            save_path (str): 若不为 None，则保存到该路径
        """
        # 使用支持日文的字体（确保路径正确）
        font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
        font = ImageFont.truetype(font_path, 36)

        # 读取图像（OpenCV读取 → RGB → PIL）
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)

        flag = True
        count = 0
        
        # 框和文字
        for box, label in zip(boxes, entities):
            count += 1
            if isinstance(label, int):
                label = kanjize.number2kanji(label)
            x1, y1, x2, y2 = map(int, box)

            # ✅ 用 PIL 画框
            for i in range(thickness):  # thicknessを表現
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
                
            # 切割图像
            if cropped:
                print("cropped.")
                cropped_image = image[y1:y2, x1:x2]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                cv2.imwrite(self.save_path + "cropped_" + str(count) + ".jpg", cropped_image)
                
            # 用Pillow画文字背景（计算文字尺寸）
            text_width, text_height = draw.textsize(label, font=font)
            text_x = x1
            text_y = y1 - text_height - 4 if y1 - text_height - 4 > 0 else y1 + 4

            if flag:
                # 标签背景
                draw.rectangle(
                    [(text_x, text_y), (text_x + text_width, text_y + text_height)],
                    fill=(color[0], color[1], color[2])
                )
                # 写文字
                draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))
            else:
                # 标签背景
                draw.rectangle(
                    [(text_x - x1 + x2 - text_width, text_y), (text_x - x1 + x2, text_y + text_height)],
                    fill=(color[0], color[1], color[2])
                )
                draw.text((text_x - x1 + x2 - text_width, text_y), label, font=font, fill=(255, 255, 255))

            flag = not flag

        # 转回BGR保存
        image_result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        if save_path:
            cv2.imwrite(save_path, image_result)
    
    def draw_res(self, id, title, Pred, GT):
        resource_filename = self.resource_path + id + ".jpg"
        npz_filename = self.npz_path + id + ".jpg.npz"
        anotation_filename = self.anaotation_path + id + ".xml"
        
        # original picture
        self.draw_boxes_and_label_on_image(resource_filename, boxes=[], color=(255,0,0), save_path=self.save_path + id + ".jpg", cropped=True)
        
        
        # 读取 xml（PASCAL VOC）
        tree = ET.parse(anotation_filename)
        root = tree.getroot()
        
        gt_boxes = []
        gt_labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            box = [
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text)
            ]
            gt_boxes.append(box)
            gt_labels.append(label)

        # 读取 npz（VinVL 风格）
        npz = np.load(npz_filename)
        npz_boxes = npz['bounding_boxes']  # shape: (N, 4), format: x1, y1, x2, y2
        print("len:", len(npz_boxes))
        # self.draw_boxes_on_image(resource_filename, boxes=npz_boxes, color=(255,0,0), save_path=self.save_path + id + "_detected.jpg")
        self.draw_boxes_and_label_on_image(resource_filename, entities=range(len(npz_boxes)), boxes=npz_boxes, color=(0,0,255), save_path=self.save_path + id + "_detected.jpg", cropped=True)
        
        # 提取所有中括号内的内容
        # Original
        matches = re.findall(r'\[.*?\]', GT)
        arrays = [ast.literal_eval(m) for m in matches]
        boxes = []
        for array in arrays:
            for a in array:
                boxes.append(npz_boxes[a])
                
        # 使用正则提取每个实体的第一个字段
        # True
        entities = re.findall(r'\(([^,]+?)\s*,', GT)
        self.draw_boxes_and_label_on_image(resource_filename, entities=entities, boxes=boxes, color=(255,0,0), save_path=self.save_path + id + "_true.jpg")
        
        # Pred
        boxes = []
        matches = re.findall(r'\[.*?\]', Pred)
        arrays = [ast.literal_eval(m) for m in matches]
        print(arrays)
        for array in arrays:
            boxes.append(npz_boxes[array[0]])
            
        # 使用正则提取每个实体的第一个字段
        entities = re.findall(r'\(([^,]+?)\s*,', Pred)
        self.draw_boxes_and_label_on_image(resource_filename, entities=entities, boxes=boxes, color=(0,255,0), save_path=self.save_path + id + "_pred.jpg")

    

def main():
    id = "NDL-112-07-091"
    title = "「なでしこ権　河原崎権十郎」 "
    Pred = "Pred: (なでしこ権 , [6] , <<役目>> ) (河原崎権十郎 , [6] , <<役者>> ) "
    GT = "GT: (なでしこ権 , [6, 10] , <<役目>> ) (河原崎権十郎 , [6, 10] , <<役者>> ) "
    
    # id = "Ebi0001(03)"
    # title = "「照手姫　尾上多賀之丞」「鬼瓦銅八　市川団升」"
    # Pred = "Pred: (照手姫 , [13] , <<役目>> ) (尾上多賀之丞 , [18] , <<役者>> ) (鬼瓦銅八 , [13] , <<役目>> ) (市川団升 , [13] , <<役者>> ) "
    # GT = "GT: (照手姫 , [5] , <<役目>> ) (尾上多賀之丞 , [5] , <<役者>> ) (鬼瓦銅八 , [2] , <<役目>> ) (市川団升 , [2] , <<役者>> ) "
    
    # id = "arcUP5146"
    # title = "「僧沙悟浄　市川寿三蔵」「八戒律　関三十郎」"
    # Pred = "Pred: (僧沙悟浄 , [12] , <<役目>> ) (市川寿三蔵 , [2] , <<役者>> ) (八戒律 , [12] , <<役目>> ) (関三十郎 , [2] , <<役者>> ) "
    # GT = "GT: (僧沙悟浄 , [18] , <<役目>> ) (市川寿三蔵 , [18] , <<役者>> ) (八戒律 , [2, 6, 10, 12] , <<役目>> ) (関三十郎 , [2, 6, 10, 12] , <<役者>> ) "
    
    # id = "NDL-107-00-045"
    # title = "「玉屋新兵衛　沢村訥升」「佐原屋半四郎　市川子団次」 "
    # Pred = "Pred: (新兵衛 , [18] , <<役目>> ) (沢村升 , [0] , <<役者>> ) (佐原屋半四郎 , [0] , <<役目>> ) (市川子団次 , [0] , <<役者>> ) "
    # GT = "GT: (新兵衛 , [0, 3] , <<役目>> ) (沢村升 , [0, 3] , <<役者>> ) (半四郎 , [1, 11, 16, 17] , <<役目>> ) (市川子団次 , [1, 11, 16, 17] , <<役者>> ) "
    
    id = "NDL-106-00-065"
    title = "「花合春之取組」_「雷電_河原崎三升」「朝日嶽_中村富十郎」"
    Pred = "Pred: (雷電 , [12] , <<替名>> ) (河原崎三升 , [3] , <<役者>> ) (中村富十郎 , [4] , <<役者>> ) "
    GT = "GT: (雷電 , [4] , <<替名>> ) (河原崎三升 , [4] , <<役者>> ) (朝日嶽 , [0] , <<替名>> ) (中村富十郎 , [0] , <<役者>> ) ) "
    
    drawer = ResDrawer()
    drawer.draw_res(id, title, Pred, GT)
    return
    
    
if __name__ == "__main__":
    main()