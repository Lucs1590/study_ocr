import cv2
import numpy as np
import tempfile

from PIL import Image
from time import time
from sklearn.cluster import KMeans


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
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

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
# to gray
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th2 = open_close(th2, cv2.MORPH_CLOSE, 1)

cv2.imwrite('tests/output.png', image)
cv2.imwrite('tests/output1.png', th2)
cv2.waitKey(0)
