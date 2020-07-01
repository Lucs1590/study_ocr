import pytesseract as ocr

from PIL import Image

phrase = ocr.image_to_string(Image.open('phrase.jpg'), lang='por')
print(phrase)
