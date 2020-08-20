import cv2
import tempfile

import numpy as np
import pytesseract as ocr

from imutils.object_detection import non_max_suppression
from PIL import Image
from sklearn.cluster import KMeans
from time import time


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    else:
        dim = (640, 640)

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # set 300 dpi
    resized = set_image_dpi(resized)

    # return the resized image
    return resized


def set_image_dpi(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)

    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))

    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(suffix='.png')
    temp_file = temp_file.name

    im_resized.save(temp_file, dpi=(300, 300))

    return np.asarray(im_resized)[:, :, ::-1]


def erode(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=3)


def dilate(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def to_kmeans(img, clusters):
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=clusters)
    clt.fit(img)
    hist = centroid_histogram(clt)
    colors = sort_colors(hist, clt.cluster_centers_)
    return colors


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def sort_colors(hist, centroids):
    aux = {}
    for (percent, color) in zip(hist, centroids):
        aux[tuple(color.astype("uint8").tolist())] = percent
    aux = sorted(aux.items(), key=lambda x: x[1], reverse=True)
    return aux


def remove_lines(image, colors):
    # Color changes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_val, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Making kernels
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    # Detect lines
    detected_h_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detected_v_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Remove lines
    h_cnts = cv2.findContours(
        detected_h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_cnts = h_cnts[0] if len(h_cnts) == 2 else h_cnts[1]
    for c in h_cnts:
        cv2.drawContours(image, [c], -1, colors[0][0], 2)

    v_cnts = cv2.findContours(
        detected_v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_cnts = v_cnts[0] if len(v_cnts) == 2 else v_cnts[1]
    for c in v_cnts:
        cv2.drawContours(image, [c], -1, colors[0][0], 2)

    cv2.imwrite('tests/thresh.png', thresh)
    # cv2.imwrite('tests/detected_lines0.png', detected_h_lines)
    # cv2.imwrite('tests/detected_lines1.png', detected_v_lines)
    cv2.imwrite('tests/image.png', image)
    cv2.waitKey()
    return image


def open_close(image, method, kernel=2):
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    result = 255 - cv2.morphologyEx(255 - image,
                                    method, repair_kernel, iterations=1)
    return result


def clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def brightness_contrast_optimization(image, alpha=1.5, beta=0):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def bin_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = open_close(th2, cv2.MORPH_CLOSE, 1)
    return image, th2


def get_size(image):
    return image.shape[0], image.shape[1]


def get_ratio(H, W):
    return H / float(640),  W / float(640)


def run_EAST(net, image, H, W):
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (H, W), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    print("[INFO] text detection took {:.6f} seconds".format(time() - start))
    return scores, geometry


def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)


def apply_boxes(boxes, image, ratio_height, ratio_width, H, W, padding):
    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * ratio_width)
        startY = int(startY * ratio_height)
        endX = int(endX * ratio_width)
        endY = int(endY * ratio_height)

        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))
        roi = image[startY:endY, startX:endX]

        config = ("-l por --oem 1 --psm 7")
        text = ocr.image_to_string(roi, config=config)

        results.append(((startX, startY, endX, endY), text))
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return results, image


def sort_boxes(boxes):
    sorted_text = []
    lines_values = sorted(list(set(map(lambda box: box[0][1], boxes))))
    for value in lines_values:
        words_of_line = sorted(
            filter(lambda box: box[0][1] == value, boxes),
            key=lambda box: box[0][0]
        )
        sorted_text.append(words_of_line)

    flatten_sorted_text = [item for sublist in sorted_text for item in sublist]
    return flatten_sorted_text


# read image
image = cv2.imread('tables/2tabela.png')
# removing alpha chanel
image = image[:, :, :3]
# histogram and contrast optimization
image = brightness_contrast_optimization(image, 1, 0.5)
# kmeans
colors = to_kmeans(image, 2)
# remove lines
image = remove_lines(image, colors)
# increase image 4 times
image = image_resize(image, height=image.shape[0]*4)
# closing image
image = open_close(image, cv2.MORPH_CLOSE)
# histogram and contrast optimization
image = brightness_contrast_optimization(image, 1, 0.5)
# improve sharp
image = unsharp_mask(image, (3, 3), 0.5, 1.5, 0)
# dilate
image = dilate(image, 1)
# to gray and B&W
(image, th2) = bin_image(image)
th2 = open_close(image, cv2.MORPH_CLOSE, 1)


""" EAST """
# copy image
original_image = th2.copy()
# set image size
(original_height, original_width) = get_size(th2)
# set image ratio
(ratio_height, ratio_width) = get_ratio(original_height, original_width)
# EAST resize pattern
image = image_resize(th2, height=640,  width=640)
# set image size
(H, W) = get_size(image)
# load EAST network
net = cv2.dnn.readNet('frozen_east_text_detection.pb')
# run EAST
(scores, geometry) = run_EAST(net, image, H, W)
# making rect and confidence limiar
(rects, confidences) = decode_predictions(scores, geometry, 0.5)
# removing overlaping boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
# applying bounding boxes
results, image = apply_boxes(boxes, original_image, ratio_height,
                             ratio_width, original_height, original_width, 0.06)
# sort bouding boxes
sorted_results = sort_boxes(results)

cv2.imwrite('tests/output.png', image)
cv2.waitKey(0)
