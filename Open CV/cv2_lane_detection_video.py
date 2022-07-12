import cv2
import numpy as np
from moviepy.editor import VideoFileClip


blur_ksize = 5


canny_lth = 50
canny_hth = 150


rho = 1
theta = np.pi / 180
threshold = 15
min_line_len = 40
max_line_gap = 20


def process_an_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    edges = cv2.Canny(blur_gray, canny_lth, canny_hth)

    rows, cols = edges.shape
    points = np.array([[(0, rows), (460, 325), (520, 325), (cols, rows)]])

    roi_edges = roi_mask(edges, points)

    drawing, lines = hough_lines(roi_edges, rho, theta,
                                 threshold, min_line_len, max_line_gap)


    draw_lanes(drawing, lines)

    result = cv2.addWeighted(img, 0.9, drawing, 0.2, 0)

    return result


def roi_mask(img, corner_points):

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, corner_points, 255)

    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


    return drawing, lines


def draw_lines(img, lines, color=[0, 0, 255], thickness=1):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):

    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return

 
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)

    
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2)
                                 for line in left_lines for x1, y1, x2, y2 in line]

    right_points = [(x1, y1)
                    for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + \
        [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_results = least_squares_fit(left_points, 325, img.shape[0])
    right_results = least_squares_fit(right_points, 325, img.shape[0])


    vtxs = np.array(
        [[left_results[1], left_results[0], right_results[0], right_results[1]]])

    cv2.fillPoly(img, vtxs, (0, 255, 0))

  

def clean_lines(lines, threshold):

    slope = [(y2 - y1) / (x2 - x1)
             for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def least_squares_fit(point_list, ymin, ymax):

    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]


    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit) 

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


if __name__ == "__main__":
    output = r'C:\Users\tarun\Desktop\open\t.mp4'
    clip = VideoFileClip(r"C:\Users\tarun\Desktop\open\Tesla.mp4")
    out_clip = clip.fl_image(process_an_image)
    out_clip.write_videofile(output, audio=False)
