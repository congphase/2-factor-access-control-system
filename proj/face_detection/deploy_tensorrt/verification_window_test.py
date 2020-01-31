import cv2
import numpy as np
import freq_cv

WINDOW_SIZE = (640, 360)
WINDOW_NAME = "VerificationWindow"
LEFT_HALF_2_POINTS = (WINDOW_SIZE[0]/2, WINDOW_SIZE[1]*4/5, 0, 0)
RIGHT_HALF_2_POINTS = (WINDOW_SIZE[0], WINDOW_SIZE[1]*4/5, WINDOW_SIZE[0]/2, 0)
DIMENSION_TO_RESIZE_TO = (int(WINDOW_SIZE[0]/2), int(WINDOW_SIZE[1]*4/5))


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Verification Window')


import cProfile, pstats, io


def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def lay_imgs(left_img, right_img, first_call=False):
    # border color; top, bottom, left, right
    color = [0, 0, 0]
    border = (5, 5, 5, 3)

    # left stuff
    if first_call is True:
        left_img_resized = cv2.resize(left_img, DIMENSION_TO_RESIZE_TO)
        left_img_resized = cv2.copyMakeBorder(left_img_resized, border[0], border[1], border[2], border[3], 
            cv2.BORDER_CONSTANT, value=color)

        blank_image = np.zeros(shape=[DIMENSION_TO_RESIZE_TO[1], DIMENSION_TO_RESIZE_TO[0], 3], dtype=np.uint8)

    # right stuff
    right_img_resized = cv2.resize(right_img, DIMENSION_TO_RESIZE_TO)
    right_img_resized = cv2.copyMakeBorder(right_img_resized, border[0], border[1], border[3], border[2], 
            cv2.BORDER_CONSTANT, value=color)

    # concatenate
    concat_img = cv2.hconcat([left_img_resized, right_img_resized])
    blank_image = np.zeros(shape=[int(WINDOW_SIZE[1]/5), concat_img.shape[1], 3], dtype=np.uint8)
    concat_img = cv2.vconcat([concat_img, blank_image])
    
    # show to window
    cv2.imshow(WINDOW_NAME, concat_img)
    cv2.imwrite('result.jpg', concat_img)
    
    return concat_img


def announce_decision(img, decision=False):
    if decision is False:
        cv2.putText(img, "ACCESS DENIED", (139, 346), cv2.FONT_HERSHEY_SIMPLEX, 1.5, freq_cv.RED, 3)
        cv2.imwrite('denied.jpg', img)
    else:
        cv2.putText(img, "ACCESS GRANTED", (128, 346), cv2.FONT_HERSHEY_SIMPLEX, 1.5, freq_cv.GREEN, 3)
        cv2.imwrite('granted.jpg', img)
    cv2.imshow('result', img)


@profile
def main():
    open_window(WINDOW_SIZE[0], WINDOW_SIZE[1])
    
    left_img = cv2.imread("/home/gate/lffd-dir/proj/face_detection/deploy_tensorrt/People/0/d_Pha (3).jpg", cv2.IMREAD_COLOR)
    #right_img = cv2.imread("/home/gate/lffd-dir/proj/face_detection/deploy_tensorrt/People/0/d_Pha (1).jpg", cv2.IMREAD_COLOR)
    right_img = cv2.imread("/home/gate/lffd-dir/proj/face_detection/deploy_tensorrt/People/1/d_Tai (1).jpg", cv2.IMREAD_COLOR)
    
    concat_returned = lay_imgs(left_img, right_img, first_call=True)
    announce_decision(concat_returned, decision=False)


    #cv2.imshow(WINDOW_NAME, concat_img)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()