import cv2
import numpy as np
from PIL import Image
import zbarlight

epsilon = 10 #image error sensitivity
test_sensitivity_epsilon = 10 #bubble darkness error sensitivity
answer_choices = ['A', 'B', 'C', 'D', 'E', '?'] #answer choices

#load tracking tags
tags = [cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)]

#test sheet specific scaling constants
scaling = [605.0, 835.0] #scaling factor for 8.5in. x 11in. paper
columns = [[72.0 / scaling[0], 33 / scaling[1]], [422.0 / scaling[0], 33 / scaling[1]]] #dimensions of the columns of bubbles
radius = 10.0 / scaling[0] #radius of the bubbles
spacing = [35.0 / scaling[0], 32.0 / scaling[1]] #spacing of the rows and columns

def ProcessPage(paper):
    answers = [] #contains answers
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY) #convert image to grayscale
    codes = zbarlight.scan_codes('qrcode', Image.fromarray(np.uint8(gray_paper))) #look for QR code
    corners = FindCorners(paper) #find the corners of the bubbled area

    #if we can't find the markers, return an error
    if corners is None:
        return [-1], paper, [-1]

    #calculate dimensions for scaling
    dimensions = [corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]]

    #iterate over test questions
    for k in range(0, 2): #columns
        for i in range(0, 25): #rows
            questions = []
            for j in range(0, 5): #answers
                #coordinates of the answer bubble
                x1 = int((columns[k][0] + j*spacing[0] - radius*1.5)*dimensions[0] + corners[0][0])
                y1 = int((columns[k][1] + i*spacing[1] - radius)*dimensions[1] + corners[0][1])
                x2 = int((columns[k][0] + j*spacing[0] + radius*1.5)*dimensions[0] + corners[0][0])
                y2 = int((columns[k][1] + i*spacing[1] + radius)*dimensions[1] + corners[0][1])

                #draw rectangles around bubbles
                cv2.rectangle(paper, (x1, y1), (x2, y2), (255, 0, 0), thickness=1, lineType=8, shift=0)

                #crop answer bubble
                questions.append(gray_paper[y1:y2, x1:x2])

            #find image means of the answer bubbles
            means = []

            #coordinates to draw detected answer
            x1 = int((columns[k][0] - radius*8)*dimensions[0] + corners[0][0])
            y1 = int((columns[k][1] + i*spacing[1] + 0.5*radius)*dimensions[1] + corners[0][1])

            #calculate the image means for each bubble
            for question in questions:
                means.append(np.mean(question))

            #sort by minimum mean; sort by the darkest bubble
            min_arg = np.argmin(means)
            min_val = means[min_arg]

            #find the second smallest mean
            means[min_arg] = 255
            min_val2 = means[np.argmin(means)]

            #check if the two smallest values are close in value
            if min_val2 - min_val < test_sensitivity_epsilon:
                #if so, then the question has been double bubbled and is invalid
                min_arg = 5

            #write the answer
            cv2.putText(paper, answer_choices[min_arg], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)

            #append the answers to the array
            answers.append(answer_choices[min_arg])

    #draw the name if found from the QR code
    if codes is not None:
        cv2.putText(paper, codes[0], (int(0.28*dimensions[0]), int(0.125*dimensions[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    else:
        codes = [-1]
    return answers, paper, codes

def FindCorners(paper):
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY) #convert image of paper to grayscale

    #scaling factor used later
    ratio = len(paper[0]) / 816.0

    #error detection
    if ratio == 0:
        return -1

    corners = [] #array to hold found corners

    #try to find the tags via convolving the image
    for tag in tags:
        tag = cv2.resize(tag, (0,0), fx=ratio, fy=ratio) #resize tags to the ratio of the image

        #convolve the image
        convimg = (cv2.filter2D(np.float32(cv2.bitwise_not(gray_paper)), -1, np.float32(cv2.bitwise_not(tag))))

        #find the maximum of the convolution
        corner = np.unravel_index(convimg.argmax(), convimg.shape)

        #append the coordinates of the corner
        corners.append([corner[1], corner[0]]) #reversed because array order is different than image coordinate

    #draw the rectangle around the detected markers
    for corner in corners:
        cv2.rectangle(paper, (corner[0] - int(ratio * 25), corner[1] - int(ratio * 25)),
        (corner[0] + int(ratio * 25), corner[1] + int(ratio * 25)), (0, 255, 0), thickness=2, lineType=8, shift=0)

    #check if detected markers form roughly parallel lines when connected
    if corners[0][0] - corners[2][0] > epsilon:
        return None

    if corners[1][0] - corners[3][0] > epsilon:
        return None

    if corners[0][1] - corners[1][1] > epsilon:
        return None

    if corners[2][1] - corners[3][1] > epsilon:
        return None

    return corners
