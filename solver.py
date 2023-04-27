import cv2
from ppadb.client import Client as Client
import numpy as np
import pytesseract   
from PIL import Image
from functools import reduce
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tesseractCustomConf = r'--oem 3 --psm 7 outputbase nobatch digits'
adb = Client(host='127.0.0.1', port=5037)
devices = adb.devices()

def get_greyscaleImage(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def hide_grid(img):
	thickness = 30
	color = (255, 255, 255)
	cv2.rectangle(img, (240, 0), (250, 700), color, thickness)
	cv2.rectangle(img, (470, 0), (480, 700), color, thickness)
	cv2.rectangle(img, (5, 240), (700, 235), color, thickness)
	cv2.rectangle(img, (5, 480), (700, 470), color, thickness)

def crop_image(img, x, y, width, height):
	return img[y:y+height, x:x+width]

def get_candidates(search_space):
    return np.setdiff1d(np.arange(1, 10), reduce(np.union1d, search_space))


def solve(board):
    missing = get_missing(board)

    if not had_missing(missing):
        return True

    missing_col = get_missing_col(missing)
    missing_row = get_missing_row(missing)

    search_space = (
            get_col(board, missing_col),
            get_row(board, missing_row),
            get_square(board, missing_col, missing_row)
        )

    for candidate in get_candidates(search_space):
        board[missing_row, missing_col] = candidate
        if solve(board):
            return True

    board[missing_row, missing_col] = 0
    return False


def get_col(board, idx):
    return board[:, idx].reshape(9)


def get_row(board, idx):
    return board[idx, :].reshape(9)


def get_square(board, col, row):
    col = col // 3 * 3
    row = row // 3 * 3
    return board[row:row+3, col:col+3].reshape(9)


def get_missing(board):
    return np.where(board == 0)


def had_missing(missing):
    return len(missing[0])


def get_missing_col(missing):
    return missing[1][0]


def get_missing_row(missing):
    return missing[0][0]


_numbers_ = [1,2,3,4,5,6,7,8,9]
def sendAdbCOmmand(number, row, column):
    _numbers = np.array([[1,2,3,4,5],[6,7,8,9,0]])
    _row_, _column_ = np.where(_numbers == number)
    _row_ = _row_[0]
    _column_ = _column_[0]
    print(f'({row},{column}),{number}')
    device.shell(f"input tap {75*(column)} {300 + 75*(row-1)}")  #taping on grid
    device.shell(f"input tap {100 +124*(_column_)} {1140 + 100*(_row_)}")  #selectiong numbers

if len(devices) == 0:
    print('No devices attached')
    quit()
device = devices[0]

image = device.screencap()
with open('test.png', 'wb') as f:
	f.write(image)
image = cv2.imread('test.png')

#Image pre-processing for OCR
cropedImage = crop_image(image, 0, 250, 720, 720)
image = thresholding(get_greyscaleImage(cropedImage))
hide_grid(image)

cv2.imwrite('cropedImage.jpg', image)
image = cv2.imread('cropedImage.jpg')
numbers = np.zeros((9,9), dtype=int)
unsolved_numbers = np.zeros((9,9), dtype=int)
print('Processing For OCR Please Wait...!')
for row in range(9):
	newY = 80*row
	for column in range(9):
		if column == 8:
			cropedImageForColumn = crop_image(image, 640, newY, 80, 80)
		else:
			cropedImageForColumn = crop_image(image, (80*column), newY, 100, 80)
		ocr_result = pytesseract.image_to_string(cropedImageForColumn, config=tesseractCustomConf)
		if ocr_result == '':
			numbers[row, column] = 0
			unsolved_numbers[row, column] = 0
		else:
			numbers[row, column] = int(ocr_result)
			unsolved_numbers[row, column] = int(ocr_result)

board = numbers

solve(board)
numbersToBePlotted=(board - unsolved_numbers)
print(board)
for number in _numbers_:
    print(f'#{number}')
    indice = []
    for row in range(1,10):
        for column in range(1,10):
            if int(numbersToBePlotted[row-1, column-1]) == number:
                indice.append((row,column))
    for _row, _column in indice:
        sendAdbCOmmand(number,_row,_column)
         
            