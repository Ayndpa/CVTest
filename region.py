import cv2
import numpy as np

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_list

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        roi_list.append((ix, iy, x, y))
        cv2.imshow('image', img)

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

def main():
    global img, drawing, roi_list
    img = cv2.imread('1.jpg')
    if img is None:
        print("Error: Could not load image.")
        return

    drawing = False
    roi_list = []

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    for roi in roi_list:
        black_border = find_black_borders(roi)
        print(f"Black border coordinates for ROI {roi}: ({black_border[0]}, {black_border[1]}, {black_border[2]}, {black_border[3]})")
        cv2.rectangle(img, (black_border[0], black_border[1]), (black_border[2], black_border[3]), (0, 0, 255), 2)

    cv2.imshow('Final Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for i, roi in enumerate(roi_list):
    #     black_border = find_black_borders(roi)
    #     cropped_img = img[black_border[1]:black_border[3], black_border[0]:black_border[2]]
    #     cv2.imwrite(f'cropped_image_{i}.jpg', cropped_img)

if __name__ == "__main__":
    main()