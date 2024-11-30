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
reference_pts2 = None  # 标准目标四点
output_size = (4961, 3508)  # 输出图像的尺寸 (宽, 高)
is_reference_processed = False  # 标记是否已处理参考图像


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_list, display_width, display_height, img

    # 将显示坐标转换为原始图像坐标
    if display_height == 0:
        return  # 防止除以零
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
    try:
        _, _, w, h = cv2.getWindowImageRect('image')
    except:
        w, h = display_width, display_height  # 默认值
    if w < 1:
        w = 1
    if h < 1:
        h = 1

    aspect_ratio = img.shape[1] / img.shape[0]
    if w / h > aspect_ratio:
        display_height = h
        display_width = int(h * aspect_ratio)
    else:
        display_width = w
        display_height = int(w / aspect_ratio)
    resized_img = cv2.resize(img, (display_width, display_height))
    cv2.imshow('image', resized_img)


def sort_points(pts):
    """
    按顺序排列四个点：左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype="float32")

    # 将点按照它们的和进行排序
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    # 计算差值并排序
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def process_image(img_path, img_files, use_first_rois=False):
    global img, drawing, roi_list, display_width, display_height, first_img_rois
    global reference_pts2, is_reference_processed, sorted_center_points_ref

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
        print(f"请在参考图像 {img_path} 上选择4个ROI并按 'q' 键确认。")
        while True:
            update_display_size()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if len(roi_list) != 4:
                    print("请确保选择了4个ROI。")
                    continue
                break
        first_img_rois = roi_list.copy()
    else:
        roi_list = first_img_rois.copy()

    cv2.destroyAllWindows()

    black_borders_list = []  # 用于存储黑块的坐标

    for idx, roi in enumerate(roi_list):
        black_border = find_black_borders(roi)
        black_borders_list.append(black_border)  # 保存每个ROI的黑边坐标
        center_x = (black_border[0] + black_border[2]) / 2
        center_y = (black_border[1] + black_border[3]) / 2
        print(
            f"Image: {os.path.basename(img_path)}, ROI {idx + 1} Black border coordinates: ({black_border[0]}, {black_border[1]}, {black_border[2]}, {black_border[3]})")
        print(f"Center coordinates: ({center_x}, {center_y})")
        cv2.rectangle(img, (black_border[0], black_border[1]), (black_border[2], black_border[3]), (0, 0, 255), 2)
        cv2.circle(img, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

    # 如果有四个ROI，进行透视变换
    if len(roi_list) == 4:
        # 计算每个ROI的中心点
        center_points = [((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2) for bb in black_borders_list]
        center_points = np.array(center_points, dtype="float32")

        # 按顺序排列点
        sorted_center_points = sort_points(center_points)

        if not use_first_rois:
            # 对参考图像，定义目标点pts2为标准尺寸的四个角
            reference_pts2 = np.array([
                [0, 0],
                [output_size[0] - 1, 0],
                [output_size[0] - 1, output_size[1] - 1],
                [0, output_size[1] - 1]
            ], dtype="float32")
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(sorted_center_points, reference_pts2)
            # 应用透视变换
            corrected_img = cv2.warpPerspective(img, M, output_size)
            # 保存对齐后的参考图像
            filename = os.path.basename(img_path)
            cv2.imwrite(os.path.join('trimimage', filename), corrected_img)
            print(f"参考图像 {filename} 已保存到 'trimimage' 目录。")
            is_reference_processed = True
            sorted_center_points_ref = sorted_center_points
        else:
            if not is_reference_processed:
                print("参考图像尚未处理，请先处理参考图像。")
                return

            # 使用参考图像的中心点和当前图像的中心点来计算变换
            M = cv2.getPerspectiveTransform(sorted_center_points, sorted_center_points_ref)
            # 获取参考图像，应用透视变换到参考图像上，以匹配当前图像的视角
            ref_img_path = img_files[0]
            ref_img = cv2.imread(ref_img_path)

            # 进行两次透视变换，以达到与参考图片相同的变换。
            warped_ref_img = cv2.warpPerspective(ref_img, M, output_size)

            # 对已经进行一次变换的参考图像再次进行透视变换，获得变换矩阵M2
            M2 = cv2.getPerspectiveTransform(sorted_center_points_ref, reference_pts2)
            warped_ref_img = cv2.warpPerspective(warped_ref_img, M2, output_size)

            # 使用参考图像经过两次变换后的变换矩阵对当前图像进行变换
            corrected_img = cv2.warpPerspective(img, M, output_size)
            corrected_img = cv2.warpPerspective(corrected_img, M2, output_size)

            # 保存对齐后的图像
            filename = os.path.basename(img_path)
            cv2.imwrite(os.path.join('trimimage', filename), corrected_img)
            print(f"图像 {filename} 已对齐并保存到 'trimimage' 目录。")
    else:
        print(f"图像 {img_path} 未检测到4个ROI，跳过对齐。")


def main():
    # 创建trimimage目录（如果不存在）
    if not os.path.exists('trimimage'):
        os.makedirs('trimimage')

    # 获取rawimage目录中所有图像文件的列表
    img_dir = 'rawimage'
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if
                 f.lower().endswith('.jpg') or f.lower().endswith('.png')]

    if not img_files:
        print("rawimage目录中没有找到图像文件。")
        return

    # 处理第一张图像以选择ROI
    process_image(img_files[0], img_files=img_files, use_first_rois=False)

    # 使用相同的ROI处理其余图像
    for img_file in img_files[1:]:
        process_image(img_file, img_files=img_files, use_first_rois=True)


if __name__ == "__main__":
    main()
