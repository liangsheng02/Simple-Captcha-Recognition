# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:24:40 2018

@author: Administrator
"""

import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

math_symbol = ['+','-','*','=']
number = ['0','1','2','3','4','5','6','7','8','9']
#alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#Chinese =
gen_char_set = math_symbol + number

def gen_captcha_text_and_image(size=(160, 60),
                               chars=gen_char_set,
                               img_type="PNG",
                               bg_color=(255, 255, 255),
                               fg_color=(0, 0, 0),
                               font_size=18,
                               font_type="arial.ttf",
                               length=5,
                               draw_lines=True,
                               n_line=(2, 4),
                               line_color=(0,0,0),
                               draw_points=True,
                               point_chance = 2,
                               point_color=(0,0,0),
                               transform=True):
    """
    @todo: 生成验证码图片
    @param size: 图片的大小，格式（宽，高），默认为(160, 60)
    @param chars: 允许的字符集合，格式为列表
    @param img_type: 图片保存的格式，默认为PNG
    @param bg_color: 背景颜色，默认为白色
    @param fg_color: 验证码字符颜色，默认为黑色
    @param font_size: 验证码字体大小
    @param font_type: 验证码字体，默认为 arial.ttf
    @param length: 验证码字符个数最大值，默认为6
    @param draw_lines: 是否划干扰线
    @param n_lines: 干扰线的条数范围，格式元组，默认为(2, 4)，只有draw_lines为True时有效
    @param line_color: 干扰线颜色，默认为黑色
    @param draw_points: 是否画干扰点
    @param point_chance: 干扰点出现的概率，大小范围[0, 100]
    @param transform: 是否创建扭曲
    @param point_color: 干扰点颜色，默认为黑色
    @return: [0]: 验证码图片中的字符串
    @return: [1]: PIL Image实例
    """
    width, height = size # 宽， 高
    captcha_image = Image.new("RGB", size, bg_color) # 创建图形
    draw = ImageDraw.Draw(captcha_image) # 创建画笔
    
    def random_captcha_text():
        """
        生成给定长度的字符串,返回列表
        """
        captcha_text = []
        for i in range(length):
            c = random.choice(gen_char_set+[''])
            captcha_text.append(c)
        return captcha_text
    
    def create_lines():
        '''绘制干扰线'''
        line_num = random.randint(*n_line) # 干扰线条数
        for i in range(line_num):
            # 起始点
            begin = (random.randint(0, size[0]), random.randint(0, size[1]))
            #结束点
            end = (random.randint(0, size[0]), random.randint(0, size[1]))
            draw.line([begin, end], fill=line_color)
    
    def create_points():
        '''绘制干扰点'''
        chance = min(100, max(0, int(point_chance))) # 大小限制在[0, 100]
        for w in range(width):
            for h in range(height):
                tmp = random.randint(0, 100)
                if tmp > 100 - chance:
                    draw.point((w, h), fill=point_color)
    
    def transform(img=captcha_image):
        '''创建扭曲'''
        # 图形扭曲参数
        params = [1 - float(random.randint(1, 2)) / 100,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 10)) / 100,
                  float(random.randint(1, 2)) / 500,
                  0.001,
                  float(random.randint(1, 2)) / 500
                  ]
        img = img.transform(size, Image.PERSPECTIVE, params) # 创建扭曲
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE) # 滤镜，边界加强(阈值更大)
        return img
    
    def create_strs():
        '''绘制验证码'''
        c_chars = random_captcha_text()
        strs = ' %s ' % ' '.join(c_chars) # 每个字符前后以空格隔开
        font = ImageFont.truetype(font_type, font_size)
        font_width, font_height = font.getsize(strs)
        draw.text(((width - font_width) / 3, (height - font_height) / 3),
                  strs, font=font, fill=fg_color)
        return c_chars
    
    if draw_lines:
        create_lines()
    if draw_points:
        create_points()
    captcha_text = create_strs()
    if transform:
        captcha_image = transform()
    
    
    return captcha_text, captcha_image

if __name__ == "__main__":
    captcha_text, captcha_image = gen_captcha_text_and_image()
    captcha_image.save("validate.PNG", "PNG")
    print(captcha_text)
    pass