import cv2
import numpy as np

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_list, display_width, display_height

    # Convert display coordinates to original image coordinates
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
    x1, y1, x2, y2 = roi
    roi_img = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:  # Ignore small contours
            min_x = min(min_x, x + x1)
            min_y = min(min_y, y + y1)
            max_x = max(max_x, x + x1 + w)
            max_y = max(max_y, y + y1 + h)

    return (min_x, min_y, max_x, max_y)

def update_display_size():
    global display_width, display_height
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

def main():
    global img, drawing, roi_list, display_width, display_height
    img = cv2.imread('3.jpg')
    if img is None:
        print("Error: Could not load image.")
        return

    drawing = False
    roi_list = []
    center_points = []

    # Set the initial display size
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

    for roi in roi_list:
        black_border = find_black_borders(roi)
        center_x = (black_border[0] + black_border[2]) // 2
        center_y = (black_border[1] + black_border[3]) // 2
        center_points.append((center_x, center_y))
        print(f"Black border coordinates for ROI {roi}: ({black_border[0]}, {black_border[1]}, {black_border[2]}, {black_border[3]})")
        print(f"Center coordinates: ({center_x}, {center_y})")
        cv2.rectangle(img, (black_border[0], black_border[1]), (black_border[2], black_border[3]), (0, 0, 255), 2)
        cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)

    if len(center_points) == 4:
        # Sort points to get consistent order: top-left, top-right, bottom-right, bottom-left
        center_points = sorted(center_points, key=lambda p: (p[1], p[0]))
        if center_points[0][0] > center_points[1][0]:
            center_points[0], center_points[1] = center_points[1], center_points[0]
        if center_points[2][0] < center_points[3][0]:
            center_points[2], center_points[3] = center_points[3], center_points[2]

        pts1 = np.float32(center_points)
        pts2 = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        min_x = min(point[0] for point in center_points)
        min_y = min(point[1] for point in center_points)
        max_x = max(point[0] for point in center_points)
        max_y = max(point[1] for point in center_points)

        cropped_img = img[min_y:max_y, min_x:max_x]
        resized_img = cv2.resize(img, (4961,3508))
        cv2.imwrite('3_exc.jpg', resized_img)

    cv2.namedWindow('Final Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Final Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()