import cv2
import numpy as np
from pyzbar.pyzbar import decode
import glob
import os

drawing = False
ix, iy = -1, -1
roi_list = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_list, display_width, display_height, img

    # 将显示坐标转换为原始图像坐标
    x_orig = int(x * img.shape[1] / display_width)
    y_orig = int(y * img.shape[0] / display_height)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_orig, y_orig

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x_orig, y_orig), (0, 255, 0), 2)
            resized_img = cv2.resize(img_copy, (display_width, display_height))
            cv2.imshow('image', resized_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x_orig, y_orig), (0, 255, 0), 2)
        roi_list.append((ix, iy, x_orig, y_orig))
        resized_img = cv2.resize(img, (display_width, display_height))
        cv2.imshow('image', resized_img)

def update_display_size():
    global display_width, display_height, img
    _, _, w, h = cv2.getWindowImageRect('image')
    aspect_ratio = img.shape[1] / img.shape[0]
    if w / h > aspect_ratio:
        display_height = h
        display_width = int(h * aspect_ratio)
    else:
        display_width = w
        display_height = int(w / aspect_ratio)
    resized_img = cv2.resize(img, (display_width, display_height))
    cv2.imshow('image', resized_img)

def recognize_barcode(image, roi):
    x1, y1, x2, y2 = roi
    roi_img = image[y1:y2, x1:x2]
    barcodes = decode(roi_img)
    for barcode in barcodes:
        if barcode.type == 'CODE128':
            return barcode.data.decode('utf-8')
    return "No CODE128 barcode found"

def main():
    global img, drawing, roi_list, display_width, display_height
    img = cv2.imread('trimimage/1_exc.jpg')
    if img is None:
        print("Error: Could not load image.")
        return

    drawing = False
    roi_list = []

    # 设置初始显示大小
    display_width = 800
    display_height = int(display_width * img.shape[0] / img.shape[1])

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        update_display_size()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(roi_list) == 0:
        print("No ROI selected.")
        return

    # 识别1_exc.jpg中的条形码信息
    barcode_info = recognize_barcode(img, roi_list[0])
    print(f"1_exc.jpg中的准考证号为：{barcode_info}")

    # 创建testok文件夹
    if not os.path.exists('testok'):
        os.makedirs('testok')

    # 识别其他*_exc.jpg文件中的条形码信息并另存
    for file in glob.glob('trimimage/*_exc.jpg'):
        # if file == '1_exc.jpg':
        #     continue
        img = cv2.imread(file)
        if img is None:
            print(f"Error: Could not load image {file}.")
            continue
        barcode_info = recognize_barcode(img, roi_list[0])
        print(f"{file}中的准考证号为：{barcode_info}")
        new_filename = os.path.join('testok', f"{barcode_info}.jpg")
        cv2.imwrite(new_filename, img)

if __name__ == "__main__":
    main()