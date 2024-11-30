import cv2
import numpy as np
import os

# 全局变量
drawing = False  # 是否正在绘制
roi_list = []  # 保存感兴趣区域（ROI）的列表
display_width = 800  # 显示窗口的宽度
display_height = 0  # 显示窗口的高度
img = None  # 当前图像
first_img_rois = []  # 第一张图像的ROI列表


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_list, display_width, display_height

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


def find_black_borders(roi):
    global img
    x1, y1, x2, y2 = roi
    roi_img = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:  # 忽略小轮廓
            min_x = min(min_x, x + x1)
            min_y = min(min_y, y + y1)
            max_x = max(max_x, x + x1 + w)
            max_y = max(max_y, y + y1 + h)

    # 确保黑边在图像边界内
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(img.shape[1], max_x)
    max_y = min(img.shape[0], max_y)

    return (min_x, min_y, max_x, max_y)


def update_display_size():
    global display_width, display_height, img
    _, _, w, h = cv2.getWindowImageRect('image')
    if w < 1: w = 1
    if h < 1: h = 1

    aspect_ratio = img.shape[1] / img.shape[0]
    if w / h > aspect_ratio:
        display_height = h
        display_width = int(h * aspect_ratio)
    else:
        display_width = w
        display_height = int(w / aspect_ratio)
    resized_img = cv2.resize(img, (display_width, display_height))
    cv2.imshow('image', resized_img)


def process_image(img_path, use_first_rois=False):
    global img, drawing, roi_list, display_width, display_height, first_img_rois
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}.")
        return

    drawing = False
    roi_list = []

    # 设置初始显示尺寸
    display_width = 800
    display_height = int(display_width * img.shape[0] / img.shape[1])

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', draw_rectangle)

    if not use_first_rois:
        while True:
            update_display_size()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        first_img_rois = roi_list.copy()
    else:
        roi_list = first_img_rois

    cv2.destroyAllWindows()

    black_borders_list = []  # 用于存储黑块的坐标

    for roi in roi_list:
        black_border = find_black_borders(roi)
        black_borders_list.append(black_border)  # 保存每个ROI的黑边坐标
        center_x = (black_border[0] + black_border[2]) // 2
        center_y = (black_border[1] + black_border[3]) // 2
        print(
            f"Black border coordinates for ROI {roi}: ({black_border[0]}, {black_border[1]}, {black_border[2]}, {black_border[3]})")
        print(f"Center coordinates: ({center_x}, {center_y})")
        cv2.rectangle(img, (black_border[0], black_border[1]), (black_border[2], black_border[3]), (0, 0, 255), 2)
        cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)

    if len(roi_list) == 4:
        # 对点进行排序以获得一致的顺序：左上，右上，右下，左下
        center_points = [(black_border[0] + (black_border[2] - black_border[0]) // 2,
                          black_border[1] + (black_border[3] - black_border[1]) // 2) for black_border in
                         black_borders_list]
        center_points = sorted(center_points, key=lambda p: (p[1], p[0]))
        if center_points[0][0] > center_points[1][0]:
            center_points[0], center_points[1] = center_points[1], center_points[0]
        if center_points[2][0] < center_points[3][0]:
            center_points[2], center_points[3] = center_points[3], center_points[2]

        # 计算裁切区域
        min_x = min(point[0] for point in center_points)
        min_y = min(point[1] for point in center_points)
        max_x = max(point[0] for point in center_points)
        max_y = max(point[1] for point in center_points)

        pts1 = np.float32(center_points)
        pts2 = np.float32([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        corrected_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        cropped_img = corrected_img[min_y:max_y, min_x:max_x]

        resized_img = cv2.resize(cropped_img, (4961, 3508))
        filename = os.path.basename(img_path)
        cv2.imwrite(f'trimimage/{filename}', resized_img)


def main():
    # 创建trimimage目录（如果不存在）
    if not os.path.exists('trimimage'):
        os.makedirs('trimimage')

    # 获取rawimage目录中所有图像文件的列表
    img_dir = 'rawimage'
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # 处理第一张图像以选择ROI
    process_image(img_files[0])

    # 使用相同的ROI处理其余图像
    for img_file in img_files[1:]:
        process_image(img_file, use_first_rois=True)


if __name__ == "__main__":
    main()