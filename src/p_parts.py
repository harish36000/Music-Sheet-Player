import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_disc(size,bg=0):
    size=int(size)
    if(size%2==0):
        size+=1
    K=np.ones((size,size))*bg
    radius=int(size//2)
    u,v=np.meshgrid(np.arange(size),np.arange(size))
    idx=(u-radius)**2+(v-radius)**2<=radius**2
    K[idx]=1
    # plt.imshow(K,cmap='gray')
    return K

def struc_slide(img,kernel,method=0,pad_img=True,round=True,ignore=-1):
    K=kernel.shape[0]
    # pad=(K-1)//2
    fg=img.max()
    img=img.astype('float64')
    out=np.zeros_like(img)
    if fg==0:
        return out
    if kernel.shape[1]%2==1:
        pad_x_min=(kernel.shape[1]-1)//2
        pad_x_max=pad_x_min
    else:
        pad_x_max=(kernel.shape[1])//2
        pad_x_min=pad_x_max-1
    if kernel.shape[0]%2==1:
        pad_y_min=(kernel.shape[0]-1)//2
        pad_y_max=pad_y_min
    else:
        pad_y_max=(kernel.shape[0])//2
        pad_y_min=pad_y_max-1
    #The input is padded with zeros on edges, so output size is same as input
    padding=((pad_y_min,pad_y_max),(pad_x_min,pad_x_max))

    if(pad_img):
        img=np.pad(img,padding)

    idx=kernel!=ignore
    for y in range(pad_y_min,img.shape[0]-pad_y_max):
        for x in range(pad_x_min,img.shape[1]-pad_x_max):
            sub_img=img[y-pad_y_min:y+pad_y_max+1,x-pad_x_min:x+pad_x_max+1]//fg
            prod=np.zeros_like(kernel)
            
            prod[idx]=(sub_img[idx]==kernel[idx])
            if method==0:
                out[y-pad_y_min,x-pad_x_min]=fg*((prod[idx]).sum()==(len(kernel[idx]))).astype('int')
            else:
                out[y-pad_y_min,x-pad_x_min]=fg*(prod[idx].sum()>0).astype('int')
    return out if not round else np.round(out)

def seg_mask(img_col_inv):
    num_peak_lst=[]
    idx_start=3
    max_k=img_col_inv.shape[0]
    for k_size in range(idx_start,max_k):
        k_ver=np.ones((k_size,1))
        eroded_ver=struc_slide(img_col_inv,k_ver)
        img_col_diff=np.abs(np.diff(eroded_ver,axis=0))
        num_peaks=img_col_diff.sum()
        # peak_diff=num_peaks-prev_peaks
        num_peak_lst.append(num_peaks)

    num_peak_arr=np.array(num_peak_lst)
    peak_diff=np.abs(np.diff(num_peak_arr,axis=0))
    k_size_opt=peak_diff.argmax()+2*idx_start
    # print(peak_diff.argmax())
    k_ver=np.ones((k_size_opt,1))
    eroded_ver=struc_slide(img_col_inv,k_ver)
    # plt.plot(num_peak_arr)
    # plt.plot(eroded_ver)
    mask_seg=eroded_ver.reshape(-1)==255
    mask_seg=mask_seg.astype('int')
    return mask_seg

def label_img(img,shaff_dist):
    kernel=make_disc(shaff_dist//2,bg=0).astype('uint8')
    # dilate=struc_slide(img,kernel,method=1).astype('uint8')
    # dilate=struc_slide(dilate,kernel,method=1).astype('uint8')
    dilate=cv2.morphologyEx(img,cv2.MORPH_DILATE,kernel,iterations=2)
    # closed=struc_slide(dilate,kernel,method=0).astype('uint8')
    # plt.imshow(closed,cmap="gray")
    # plt.imshow(dilate,cmap="gray")
    analysis = cv2.connectedComponentsWithStats(dilate,4,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    return totalLabels,label_ids,values,centroid


def segment_notes(img_th_inv,mask_seg):
    idx1=-1
    idx2=-1
    # print(mask_seg)
    mask_seg_mid=np.zeros_like(mask_seg)
    # midpoints=[]
    # for i in range(mask_seg.shape[0]-1):
    #     if((i==0 and mask_seg[i]>0) or (mask_seg[i]-mask_seg[i+1]<0)):
    #         idx1=i
    #     elif i+1==mask_seg.shape[0] or  (mask_seg[i]-mask_seg[i+1]>0):
    #         idx2=i
    #     if idx1!=-1 and idx2!=-1:
    #         # print(idx1,idx2)
    #         mask_seg_mid[(idx1+idx2)//2]=1
    #         midpoints.append((idx1+idx2)//2)
    #         idx1=-1
    #         idx2=-1
    pts_o_ch=np.diff(mask_seg.astype('int'))
    pts_pve=np.where(pts_o_ch>0)[0]
    pts_nve=np.where(pts_o_ch<0)[0]
    # idx_o_ch=np.where(pts_o_ch==1)[0]
    if(pts_pve.shape<pts_nve.shape):
        pts_pve=np.resize(pts_nve.shape)
        pts_pve[1:]=pts_pve[0:-1]
        pts_pve[0]=0
    if(pts_pve.shape>pts_nve.shape):
        pts_nve=np.resize(pts_pve.shape)
        pts_nve[-1]=img_th_inv.shape[0]
    # midpoints=(idx_o_ch[0:-1]+idx_o_ch[1:])//2
    midpoints=(pts_pve+pts_nve)//2
    # print(midpoints)
    img_segments=[]
    # orig_img_segments=[]
    # img_segments.append()
    for i in  range(len(midpoints)):
        # if i==0:
        #     tmp_img=img_th_inv[0:midpoints[i]].copy()
        if i==len(midpoints)-1:
            tmp_img=img_th_inv[midpoints[i]:img_th_inv.shape[0]].copy()
            # tmp_img1=(255-img_th)[midpoints[i]:img_th_inv.shape[0]].copy()
        else:
            tmp_img=img_th_inv[midpoints[i]:midpoints[i+1]].copy()
            # tmp_img1=(255-img_th)[midpoints[i]:midpoints[i+1]].copy()
        img_segments.append(tmp_img)
        # orig_img_segments.append(tmp_img1)
    # plt.imshow(img_segments[0],cmap="gray")
    # print(midpoints)
    return img_segments#,orig_img_segments

# DETECT Quarter notes:

def qtr_note_det(orig_img_segments,label_segments,values,shaff_dist,dic_lbl):
    kernel=make_disc(shaff_dist,bg=0).astype('uint8')
    note_wo=[]
    out_segs=[]
    
    # dic_lbl={'s':-1,
    #         'q':-1,
    #         'e':-1}
    for i,img in enumerate(orig_img_segments):
        # test_segment1=struc_slide(img,kernel)
        # note_wo_hole=struc_slide(test_segment1,kernel,method=1)
        note_wo_hole=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        if i==0:
            plt.imshow(note_wo_hole)
        # note_wo_hole=cv2.dilate(test_segment1,kernel).astype('uint8')
        # plt.imshow(note_wo_hole,cmap='gray')
        # DETECT CIRCULAR ELEMENTS
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200


        # Filter by Area.
        params.filterByArea = False
        params.minArea = 1500

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.1
        params.maxConvexity = 1


        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        # print(np.unique(note_wo_hole[1]))
        # plt.imshow(255-note_wo_hole[1],cmap='gray')
        # params.filterByArea = True
        # params.minArea = 1
        # params.maxInertiaRatio = 0.9
        detector=cv2.SimpleBlobDetector_create(params)
        keypoints=detector.detect(255-note_wo_hole)
        blank = np.zeros((0, 0))
        # blank = np.zeros((shaff_dist, 1))
        blobs = cv2.drawKeypoints(np.zeros_like(note_wo_hole), keypoints,0, (255, 255, 255)).max(axis=2)
        blobs[blobs!=0]=255
        note_wo.append(blobs)


        notes_wo_hole_lbl=(blobs!=0)*label_segments[i]
        # print(np.unique(notes_wo_hole_lbl))
        # MARK THE LABELS::
        label_segs=label_segments[i].copy()
        # print(np.unique(label_segments[1]))
        # max_area=0
        # for label in np.unique(label_segments[i]):
        #     area=values[label,cv2.CC_STAT_AREA]
        #     if(area>max_area):
        #         max_area=area
        # print(max_area)
        area_note=(shaff_dist**2)*32
        for label in np.unique(label_segments[i]):
            # print(label)
            area=values[label,cv2.CC_STAT_AREA]
            # if (label in np.unique(notes_wo_hole_lbl)):
            #     print(label,area)
            if label not in np.unique(notes_wo_hole_lbl) or label == 0:
                # label_segs[label_segs==label]=0
                continue
            elif area<area_note:
                print(area,area_note)
                if 's' not in dic_lbl.keys():
                    dic_lbl['s']=label
                # if dic_lbl['s']==-1:
                # print("In HERE:",label,dic_lbl['s'])
                label_segs[label_segs==label]=dic_lbl['s']
            elif area>=area_note and area<area_note*3:
                # if dic_lbl['q']==-1:
                print(area,area_note)
                if 'q' not in dic_lbl.keys():
                    dic_lbl['q']=label
                label_segs[label_segs==label]=dic_lbl['q'] 
            else:
                # if dic_lbl['e']==-1:
                print(area,area_note)
                if 'e' not in dic_lbl.keys():
                    dic_lbl['e']=label
                label_segs[label_segs==label]=dic_lbl['e'] 
        label_segments[i]=label_segs
        # plt.imshow(label_segs)
        # out_segs.append(label_segs)

    if 'e' not in dic_lbl.keys():
        dic_lbl['e']=-1
    if 's' not in dic_lbl.keys():
        dic_lbl['s']=-1
    if 'q' not in dic_lbl.keys():
        dic_lbl['q']=-1
    return note_wo

def half_note_det_OLD(img_segments,label_segments,values,shaff_dist,dic_lbl):
    kernel=make_disc(shaff_dist,bg=0).astype('uint8')
    notes_wh=[]
    # dic_lbl={'sh':-1,'qh':-1,'eh':-1}
    for i,img in enumerate(img_segments):

        # dilated=cv2.dilate(img_segments[0],kernel)
        fill_holes=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        diff_img=fill_holes-img

        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200


        # Filter by Area.
        params.filterByArea = True

        shaff_d=shaff_dist.astype('float')
        params.maxArea = (shaff_d)**2
        params.minArea = shaff_d

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.55

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7
        params.maxConvexity = 0.9


        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.16
        params.maxInertiaRatio = 0.22

        # print(np.unique(note_wo_hole[1]))
        # plt.imshow(255-note_wo_hole[1],cmap='gray')
        # params.filterByArea = True
        # params.minArea = 1
        # params.maxInertiaRatio = 0.9
        detector=cv2.SimpleBlobDetector_create(params)
        keypoints=detector.detect(255-diff_img)
        blank = np.zeros((1, 1))
        # blank = np.zeros((shaff_dist, 1))

        blobs = cv2.drawKeypoints(np.zeros_like(diff_img), keypoints,0, (255, 255, 255)).max(axis=2)
        # note_wo.append(blobs)
        notes_wh.append(blobs)
        
        notes_w_hole_lbl=(blobs!=0)*label_segments[i]
        # plt.imshow(diff_img)
        # print(np.unique(notes_wo_hole_lbl))
        # MARK THE LABELS::
        label_segs=label_segments[i].copy()
        area_note=(shaff_dist**2)*8
        # print(np.unique(label_segments[1]))
        for label in np.unique(label_segments[i]):
            # print(label)
            area=values[label,cv2.CC_STAT_AREA]
            # if (label in np.unique(notes_wo_hole_lbl)):
            #     print(label,area)
            if label in np.unique(notes_w_hole_lbl) and label!=0:
                to_plot=np.zeros_like(label_segments[i])
                to_plot[label_segments[i]==label]=255
                # print('label:',label)
                # plt.imshow(to_plot)
            if label not in np.unique(notes_w_hole_lbl) or label == 0:
                # label_segs[label_segs==label]=0
                continue
            elif area<area_note//2:
                # if dic_lbl['sh']==-1:
                if 'wh' not in dic_lbl.keys():
                    dic_lbl['wh']=label
                # print("In HERE:",label,dic_lbl['s'])
                label_segs[label_segs==label]=dic_lbl['hh']
            # elif area>=area_note and area<area_note*3:
            #     # if dic_lbl['qh']==-1:
            #     if 'qh' not in dic_lbl.keys():
            #         dic_lbl['qh']=label
            #     label_segs[label_segs==label]=dic_lbl['qh'] 
            else:
                # if dic_lbl['eh']==-1:
                if 'hh' not in dic_lbl.keys():
                    dic_lbl['hh']=label
                label_segs[label_segs==label]=dic_lbl['hh'] 
        label_segments[i]=label_segs
        # plt.imshow(label_segs)
        # out_segs.append(label_segs)
        # break
    if 'wh' not in dic_lbl.keys():
        dic_lbl['wh']=-1
    if 'hh' not in dic_lbl.keys():
        dic_lbl['hh']=-1
    return notes_wh

def add_staff(staff_segments,staff_dist):
    staff_segments_e=[]
    for staffs in staff_segments:
        staff=staffs.copy()
        # print(staff.shape)
        idxs=np.where(staff!=0)[0]
        # print(idxs)
        if idxs.shape[0]==0:
            continue
        start_idx=max(0,idxs[0]-staff_dist)
        end_idx=min(staff.shape[0]-1,idxs[-1]+staff_dist)
        if end_idx<=start_idx:
            start_idx=0
            end_idx=staff.shape[0]
        staff[start_idx,:]=staff.max()
        staff[end_idx,:]=staff.max()
        staff_segments_e.append(staff.astype('uint8'))
    return staff_segments_e

def half_note_det(img_segments,label_segments,staff_segments_e,note_wo,values,shaff_dist,dic_lbl):
    kernel=make_disc(shaff_dist,bg=0).astype('uint8')
    notes_wh=[]
    # dic_lbl={'sh':-1,'qh':-1,'eh':-1}
    for i,img in enumerate(img_segments):

        # dilated=cv2.dilate(img_segments[0],kernel)
        fill_holes1=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        fill_holes=cv2.morphologyEx(fill_holes1,cv2.MORPH_OPEN,kernel)
        idxs=np.where(staff_segments_e[0]!=0)[0]
        end_idx=min(fill_holes.shape[0],idxs[-1])
        start_idx=max(0,idxs[0])
        if idxs.shape[0]<=1 or start_idx>=end_idx:
            idxs=np.array([0,img.shape[0]])
            start_idx=0
            end_idx=img.shape[0]
        diff_img=fill_holes[start_idx:end_idx,:]
        print(idxs.shape,start_idx,end_idx)
        # print(max(0,idxs[0]),)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200


        # Filter by Area.
        params.filterByArea = False

        shaff_d=shaff_dist.astype('float')
        params.maxArea = (shaff_d)**2
        params.minArea = shaff_d

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.55

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.7
        params.maxConvexity = 0.9


        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.55
        params.maxInertiaRatio = 0.9

        # print(np.unique(note_wo_hole[1]))
        # plt.imshow(255-note_wo_hole[1],cmap='gray')
        # params.filterByArea = True
        # params.minArea = 1
        # params.maxInertiaRatio = 0.9
        detector=cv2.SimpleBlobDetector_create(params)
        keypoints=detector.detect(255-diff_img)
        blank = np.zeros((1, 1))
        print(diff_img.shape)
        blobs = cv2.drawKeypoints(np.zeros_like(diff_img), keypoints,0, (255, 255, 255)).max(axis=2)
        # note_wo.append(blobs)
        blobs[blobs!=0]=255
        blob_a=np.zeros_like(note_wo[i])
        blob_a[start_idx:end_idx,:]=blobs
        # blob_a=(blob_a!=0)*(np.max(note_wo[i])-note_wo[i])
        
        lbl_seg=(label_segments[i])[start_idx:end_idx,:]
        notes_w_hole_lbl=(blobs!=0)*(lbl_seg)
        # plt.imshow(diff_img)
        # print(np.unique(notes_wo_hole_lbl))
        # MARK THE LABELS::
        label_segs=label_segments[i].copy()
        labels_filled=[dic_lbl['e'],dic_lbl['q'],dic_lbl['s']]
        # area_note=(shaff_dist**2)*8
        area_note=(shaff_dist**2)*32
        # print(np.unique(label_segments[1]))
        for label in np.unique(lbl_seg):
            # print(label)
            area=values[label,cv2.CC_STAT_AREA]
            # if (label in np.unique(notes_wo_hole_lbl)):
            #     print(label,area)
            if label in np.unique(notes_w_hole_lbl) and label not in labels_filled:
                to_plot=np.zeros_like(label_segments[i])
                to_plot[label_segments[i]==label]=255
                # print('label:',label)
                # plt.imshow(to_plot)
            if label not in np.unique(notes_w_hole_lbl) or label == 0 :
                # label_segs[label_segs==label]=0
                continue
            if label in labels_filled:
                idx=label_segments[i]==label
                blob_a[idx]=0
                continue
            elif area<area_note//2:
                # if dic_lbl['sh']==-1:
                if 'wh' not in dic_lbl.keys():
                    dic_lbl['wh']=label
                # print("In HERE:",label,dic_lbl['s'])
                label_segs[label_segs==label]=dic_lbl['wh']
            # elif area>=shaff_dist*128 and area<shaff_dist*768:
            #     # if dic_lbl['qh']==-1:
            #     if 'qh' not in dic_lbl.keys():
            #         dic_lbl['qh']=label
            #     label_segs[label_segs==label]=dic_lbl['qh'] 
            else:
                # if dic_lbl['eh']==-1:
                if 'hh' not in dic_lbl.keys():
                    dic_lbl['hh']=label
                label_segs[label_segs==label]=dic_lbl['hh']
        label_segments[i]=label_segs
        notes_wh.append(blob_a)
        # plt.imshow(label_segs)
        # out_segs.append(label_segs)
        # break
    if 'wh' not in dic_lbl.keys():
        dic_lbl['wh']=-1
    if 'hh' not in dic_lbl.keys():
        dic_lbl['hh']=-1
    
    return notes_wh

def make_blob_dic(label_segments,note_wo,note_wh,dic_lbl,shaff_dist):
    blob_dic={}
    blob_dic={'s':[],'e':[],'q':[],'hh':[],'wh':[]}
    blob_dic_dilated={'s':[],'e':[],'q':[],'hh':[],'wh':[]}
    note_pts_list=[]
    for i,label_segment in enumerate(label_segments):
        blob_dic['s'].append((label_segment==dic_lbl['s'])*(note_wo[i]!=0).astype('uint8'))
        blob_dic['e'].append((label_segment==dic_lbl['e'])*(note_wo[i]!=0).astype('uint8'))
        blob_dic['q'].append((label_segment==dic_lbl['q'])*(note_wo[i]!=0).astype('uint8'))
        blob_dic['hh'].append((label_segment==dic_lbl['hh'])*(note_wh[i]!=0).astype('uint8'))
        blob_dic['wh'].append((label_segment==dic_lbl['wh'])*(note_wh[i]!=0).astype('uint8'))
        # blob_dic_dilated={}
        note_pts=np.zeros_like(note_wh[i])
        kernel=make_disc(shaff_dist,bg=0).astype('uint8')
        for key in blob_dic.keys():
            # print(key)
            morph=cv2.morphologyEx((blob_dic[key])[i],cv2.MORPH_DILATE,kernel)
            # print(morph.shape)
            blob_dic_dilated[key].append(morph)
            note_pts+=blob_dic_dilated[key][i]
        note_pts_list.append(note_pts)
    return blob_dic,note_pts_list,blob_dic_dilated
    
def label_notes(note_pts_list):
    labelled_list=[]
    prev_max=0
    for note_pts in note_pts_list:
        analysis = cv2.connectedComponentsWithStats(note_pts,8,cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis
        label_ids[label_ids!=0]=label_ids[label_ids!=0]+prev_max
        labelled_list.append(label_ids)
        prev_max=np.max(label_ids)+1
    return labelled_list

def get_dic(blob_dic,labelled_list):
    dic_mega={}
    for i,label_ids in enumerate(labelled_list):
        print(np.unique(label_ids))
        for key in blob_dic.keys():
            # print(np.unique(blob_dic[key]))
            labels=np.unique((blob_dic[key][i]!=0)*label_ids)
            for label in labels:
                print(label)
                if label!=0:
                    dic_mega[label]=key
    return dic_mega