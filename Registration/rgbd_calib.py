# rgbd calib for 7-scenes dataset.
import numpy as np
import cv2 # to calibrate

# color align to depth
def color_calibration(_color): 
    color = np.copy(_color)
    height = color.shape[0]
    width = color.shape[1]
    a = 1.0/0.89 # 0.88, 0.89
    h_new = int(height*a)
    w_new = int(width*a)
    size = (w_new, h_new)  
    dr = int((h_new - height)/2)
    dc = int((w_new - width)/2)
    color_resized = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
    result = np.array(color_resized[dr:dr+480, dc:dc+640], dtype=np.uint8)
    return result
