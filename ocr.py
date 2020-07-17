import pytesseract as ocr
from PIL import Image
from time import time

start_time = time()
phrase = ocr.image_to_string(Image.open('./images/tabela.png'), lang='por')
print(phrase)
print("Execution Time:", time() - start_time)
