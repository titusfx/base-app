import cv2

# import easyocr
# # Initialize Easy OCR
# reader = easyocr.Reader(["en"])
# detections = reader.readtext(current_frame)
# bbox, text, confidence = detections[0]

from paddleocr import PaddleOCR, draw_ocr

# # Initialize OCR model
# ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Supports multiple languages, here we use English
# # Image path
# img_path = 'gameboard.jpg'
# # Perform OCR
# result = ocr.ocr(img_path)
# # Print results
# for line in result:
#     print(line)
# # Visualize the results
# image = cv2.imread(img_path)
# boxes = [elements[0] for elements in result[0]]  # Extract boxes from OCR results
# txts = [elements[1][0] for elements in result[0]]  # Extract recognized texts
# scores = [elements[1][1] for elements in result[0]]  # Extract recognition confidence


class Reader:

    def __init__(self, language) -> None:
        self.ocr = PaddleOCR(
            use_angle_cls=True, lang="en"
        )  # Supports multiple languages, here we use English

    def readtext(self, current_frame):
        # Convert the frame to RGB as PaddleOCR expects RGB format
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # Perform OCR on the frame
        ocr_result = self.ocr.ocr(frame_rgb)
        # Print OCR results
        for line in ocr_result:
            print(line, "\n")

        ocr_result = [(line[1][0], line[0]) for line in ocr_result[0]]
        return ocr_result


from abc import ABC, abstractmethod


class IReader(ABC):
    @abstractmethod
    def readtext(self, frame):
        pass


class PaddleReader(IReader):
    def __init__(self, language="en"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=language)

    def readtext(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(frame_rgb)

        ocr_results = []
        if result[0] is None:
            return []
        for line in result[0]:
            # print(line)
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            ocr_results.append((bbox, text, confidence))

        return ocr_results
