import pytesseract as ocr
import glob
import os
from PIL import Image
from time import time


caminho = "/home/brito/Documentos/Dev/ocr_tests/tables"

f = open("results1.txt", "a+")

for arquivo in glob.glob(os.path.join(caminho, "*.png")):
    start_time = time()
    phrase = ocr.image_to_string(Image.open(arquivo), lang='por')
    f.write(phrase + "\n")
    f.write("Execution Time: {}\n".format(time() - start_time))
    f.write("----------------------------------------\n")
