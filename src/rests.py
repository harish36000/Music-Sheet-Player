import numpy as np
import cv2

def extract_verticals(segments,threshold_per=0.7):
    segments=segments.copy()
    segments=np.array(segments)
    output_segments=np.zeros(segments.shape)
    for i in range(segments.shape[0]):
        rowsums=np.sum(segments,axis=0)/segments.shape[0]
        max_val=np.max(rowsums)
        threshold=threshold_per*max_val
        rowsums[rowsums<=threshold]=0
        rowsums[rowsums>0]=255
        output_segments[:,rowsums>0]=segments[:,rowsums>0]
    return output_segments

def detect_rests(segments,labels,var_threshold=0.25):
    unique_labels=np.unique(labels)
    rest_labels=[]
    for label in unique_labels:
        img1=np.zeros(labels.shape)
        if label==0:
            continue
        img1[labels==label]=segments[labels==label]
        img1_th=cv2.threshold(img1.astype('uint8'),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        img1_th=np.array(img1_th,dtype='uint8')
        img1_th=np.sum(img1_th,axis=0).astype(int)
        img1_th=img1_th[np.nonzero(img1_th)]
        if img1_th.shape[0]==0:
            continue
        mode=np.argmax(np.bincount(img1_th))
        mode_var=np.mean(np.square((img1_th-mode)/255))
        if mode_var<var_threshold:
            rest_labels.append(label)
    return rest_labels


def get_rests(img_segments,label_segments):
    number_segments=len(img_segments)
    vertical_lines=[]

    for i in range(number_segments):
        vertical_lines.append(extract_verticals(img_segments[i]))

    lines_without_verticals=[]
    for i in range(number_segments):
        lines_without_verticals.append(img_segments[i]-vertical_lines[i])
    
    rest_labels=[]

    for i in range(number_segments):
        rest_labels.append(detect_rests(lines_without_verticals[i],label_segments[i]))
    
    return rest_labels
