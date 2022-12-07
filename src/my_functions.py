import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy import ndimage

def findTiltAngle(img_edges):
  
  lines= cv2.HoughLines(img_edges, 1, np.pi/180.0, 250, np.array([]))
  for line in lines:
      rho, theta = line[0]
  theta = theta*180/np.pi

  if (theta < 0):
    
    theta = theta + 90
    
  else:
    
    theta = theta - 90
    
    return theta

def correctSkew(img_th):
    img_edges = cv2.Canny(img_th, 100, 200)
    theta = findTiltAngle(img_edges)
    img_rotated = ndimage.rotate(img_th, theta)
    return img_rotated

def getMode(a):
    val, counts = np.unique(a, return_counts=True)
    indx = np.argsort(counts)[::-1]
    val = val[indx]
    mode = val[0]
    if (mode == 0 and len(val)>1): # Makes sure that the background pixel label is not returned
        mode = val[1]

    return mode

def getClef(line, rowsums):
    clef_arr = []
    totpeaks = 1
    numpeaks = 0
    i=0

    while (i < rowsums.shape[0]):
        if (rowsums[i] > 0):
            i_entered = i
            while(rowsums[i] > 0):
                clef_arr.append(i)
                i = i+1

            if (np.abs(i - i_entered) <= (2*line.shape[0])/100):
                totpeaks = 2
            numpeaks = numpeaks+1 
            if (numpeaks >= totpeaks):
                break

        else:
            i = i+1

    return line[:, clef_arr]

def LineWidthNorm(a):
    tempsums = np.sum(a, axis=1)
    for i in range(tempsums.shape[0] - 1):
        if (tempsums[i] != 0 and tempsums[i+1] != 0):
            j = i+1
            while(tempsums[j] != 0):
                a[j] = 0
                tempsums[j] = 0
                j += 1

    return a

def lineToNoteMap(lineNumber):
    if (lineNumber==7):
        return 'c'
    elif(lineNumber==6):
        return 'E'
    elif(lineNumber == 5):
        return 'g'
    elif(lineNumber == 4):
        return 'B'
    elif(lineNumber == 3):
        return 'D'
    elif(lineNumber == 2):
        return 'F'
    elif(lineNumber == 1):
        return 'A'

def spaceToNoteMap(spaceNumber):
    if(spaceNumber == 6):
        return 'd'
    elif(spaceNumber == 5):
        return 'f'
    elif(spaceNumber == 4):
        return 'a'
    elif(spaceNumber == 3):
        return 'C'
    elif(spaceNumber == 2):
        return 'E'
    elif(spaceNumber == 1):
        return 'G'

# Takes the normalized line mask as input

def getStaffLineNums(lineMask):
    tempsum = np.sum(lineMask, axis=1)
    staffLineToRow = {}
    j = 1
    for i in range(tempsum.shape[0]):
        if (tempsum[i] != 0):
            staffLineToRow[j] = i
            j += 1

    return staffLineToRow

def NoteLineAnding(lineMask, labelledNoteMask):
    # anded_arr = np.multiply(lineMask, labelledNoteMask) // np.max(lineMask)
    anded_arr=(lineMask!=0)*labelledNoteMask
    return anded_arr.astype('uint8')

def noteIdentification(staffLinetoRow, anded_array):
    labelToNote = {} # stores the mapping
    for i in range(1, len(staffLinetoRow)+1): # iterate through staff line numbers
        curr_row = staffLinetoRow[i] # row index of current staff line
        tempsum = np.sum(anded_array[curr_row])
        if (tempsum != 0):
            uniqueLabelsInLine = np.unique(anded_array[curr_row]) # Get all unique labels in the line
            for j in range(uniqueLabelsInLine.shape[0]):
                if (uniqueLabelsInLine[j] in labelToNote.keys()): # Handles space note case
                    labelToNote.pop(uniqueLabelsInLine[j])
                    labelToNote[uniqueLabelsInLine[j]] = spaceToNoteMap(i-1)
                else: # Handles line note case
                    labelToNote[uniqueLabelsInLine[j]] = lineToNoteMap(i)

    if (0 in labelToNote.keys()):
        labelToNote.pop(0)  
    return labelToNote

def getNoteSequence(anded_array, labelToNote):
    noteString = []
    tempsum = np.sum(anded_array, axis=0)
    _, idx = np.unique(tempsum, return_index=True)
    tempsum_sorted = tempsum[np.sort(idx)]
    for i in range(tempsum_sorted.shape[0]):
        if (tempsum_sorted[i] in labelToNote.keys()):
            noteString.append(labelToNote[tempsum_sorted[i]])

    return noteString

# This function takes the Normalized lineMask
def getSegmentNotes(lineMask, labelledNoteMask,mega_dic):
    segmentNotes = []
    staffLinetoRow = getStaffLineNums(lineMask)
    anded_array = NoteLineAnding(lineMask, labelledNoteMask)
    # dic_remap, anded_array = labelRemapping(anded_array)
    anded_array=2*anded_array+1
    label2note=noteIdentification(staffLinetoRow,anded_array)
    final_mapping = noteIdentification2(mega_dic,staffLinetoRow, anded_array,label2note)

    noteString = getNoteSequence2(final_mapping)
    segmentNotes.append([noteString])

    return segmentNotes

def labelRemapping(anded_array):
    unique_labels = np.unique(anded_array)
    remap_dic = {}
    for i in range(unique_labels.shape[0]):
        remap_dic[unique_labels[i]] = 2*i + 1

    for i in remap_dic:
        if (i != 0):
            anded_array[anded_array == i] = remap_dic[i]

    return remap_dic, anded_array

def getKeySig(line, rowsums):
    keysig_arr = []
    clef_arr = []
    totpeaks = 1
    numpeaks = 0
    i=0

    while (i < rowsums.shape[0]):
        if (rowsums[i] > 0):
            i_entered = i
            while(rowsums[i] > 0):
                clef_arr.append(i)
                i = i+1

            if (np.abs(i - i_entered) <= (2*line.shape[0])/100):
                totpeaks = 2
            numpeaks = numpeaks+1 
            if (numpeaks >= totpeaks):
                break

        else:
            i = i+1

    while (i < rowsums.shape[0]):
        if (rowsums[i] > 0):
            i_entered = i
            break

        else:
            i = i+1

    return line[:, i:20], i

def getNoteKeySig(eroded_sig):
    tempsum = np.sum(eroded_sig, axis=0)
    sig = []
    i_enter2 = 100
    i=0

    while (i < tempsum.shape[0]):
        if (tempsum[i] > 0):
            while (tempsum[i]>0):
                i += 1
            i_end1 = i
            while (tempsum[i] == 0 and i < tempsum.shape[0]-1):
                i += 1
            if (tempsum[i] > 0):
                i_enter2 = i
            
            if (np.abs(i_enter2 - i_end1) < 4):
                sig.append('s')
            else:
                sig.append('f')

        else:
            i += 1
                
    return sig

def firstElem(e):
    return e[0]

def getNoteSequence2(finalMapping):
    finalMapping_sorted = finalMapping.copy()
    finalMapping_sorted.sort(key=firstElem)
    MusicString = []
    for i in range(len(finalMapping_sorted)):
        if (finalMapping_sorted[i][3] == 'wh'):
            MusicString.append([finalMapping_sorted[i][2], 8])
        elif (finalMapping_sorted[i][3] == 'hh'):
            MusicString.append([finalMapping_sorted[i][2], 4])
        elif (finalMapping_sorted[i][3] == 's'):
            MusicString.append([finalMapping_sorted[i][2], 2])
        elif (finalMapping_sorted[i][3] == 'e' or finalMapping_sorted == 'q'):
            MusicString.append([finalMapping_sorted[i][2], 1])

    return MusicString

def noteIdentification2(labelToDurationDic, staffLinetoRow, anded_array, labelToNote):
    finalMapping = []
    label_mapped=[]
    for i in range(1, len(staffLinetoRow)+1):
        curr_row = staffLinetoRow[i]
        tempsum = np.sum(anded_array[curr_row])
        if (tempsum != 0): # Check if the line has intersection with some note
            uniqueLabelsInLine = np.unique(anded_array[curr_row]) # Get all unique labels in the line
            # uniqueLabelsInLine=(uniqueLabelsInLine-1)//2
            for j in range(uniqueLabelsInLine.shape[0]):

                my_label = uniqueLabelsInLine[j]
                if my_label==1:
                    continue
                my_char = labelToNote[my_label]
                my_dur = labelToDurationDic[(my_label-1)//2]
                my_col = np.where(anded_array[curr_row]==my_label)[0][0]
                if (my_label not in label_mapped):
                    label_mapped.append(my_label)
                    finalMapping.append([my_col, my_label, my_char, my_dur])
                # print("In id", finalMapping)

    return finalMapping