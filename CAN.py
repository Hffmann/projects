import cv2
import numpy as np
import os
import pdb
from datetime import datetime



mask = int(0x03FFFF00)    # Suitable mask to extract the bits with

log_file = r'C:\Python\yesweCAN/DAF_TRUCK_000.log'

pgn_dict = {}
data = ""
function = ""

with open(log_file) as f:
    f = f.readlines()

date_time = time = [ 0 for i in range(len(f))]
post_rqst_time = [ 0 for i in range(len(f))]
format = '%H:%M:%S.%f'
temporary = datetime.strptime("00:00:00.0000", format)
k = l = j = n = post_count = p1 = p2 = q1 = q2 = r1 = r2 = s1 = s2 = t1 = t2 = test_count = test_count2 = 0
state = 0


for line in f:
    xx= line.strip().split(' ')
    my_hexdata = int(xx[3], 16)
    hex_data = hex(my_hexdata & mask)
    pgn = hex_data[:6]

    if state == 0 and xx[3] == '0x18DA3DF9' or xx[3] == '0x18DAF93D':
        test_count +=1
        '''print(xx)'''

    if state == 0 and xx[3] == '0x18DA0BF9' or xx[3] == '0x18DAF90B':
        test_count2 +=1
        print(xx)


    if xx[3] == '0x18EAFFFE' or xx[3] == "0x18EEFFF9" or xx[3] == "0x18DFFFF9": #request message starting at EAS-2 programming

        if state == 0:
            print( 'ATTENTION 3D: {}'.format(test_count))
            print( 'ATTENTION 0B: {}'.format(test_count2))

            state = 1

        xx[0] = xx[0].replace(xx[0][-5:], xx[0][-5:].replace(":", "."))
        '''
        for i in range(8):
            data += xx[i+5]
        '''

        if xx[3] == '0x18EAFFFE':
            function = "RQST"
            RQST_time = datetime.strptime(xx[0], format)

        if xx[3] == "0x18EEFFF9":
            function = "ACL"
            ACT_time = datetime.strptime(xx[0], format)
            post_rqst_time[post_count] = ACT_time - RQST_time
            post_count += 1

        if xx[3] == "0x18DFFFF9" and xx[6] == '0F':
            function = "STOP BROADCAST"

        if xx[3] == "0x18DFFFF9" and xx[6] == 'FF':
            function = "KEEP BD STATE"

        if xx[3] == "0x18DFFFF9" and xx[6] == '5F':
            function = "START BROADCAST"




        print(function, xx)

        if k != 0 or l != 0:
            print("\n \n \n**************** programming frames F9 to 3D: {}    programming frames 3D to F9: {} \n \n \n".format(k, l))

        if p1 != 0 or p2 != 0:
            print("programming frames F9 to 4D: {}    programming frames 4D to F9: {} \n".format(p1, p2))

        if q1 != 0 or q2 != 0:
            print("programming frames F9 to 27: {}    programming frames 27 to F9: {} \n".format(q1, q2))

        if r1 != 0 or r2 != 0:
            print("programming frames F9 to 25: {}    programming frames 25 to F9: {} \n".format(r1, r2))

        if s1 != 0 or s2 != 0:
            print("programming frames F9 to 00: {}    programming frames 00 to F9: {} \n".format(s1, s2))

        if t1 != 0 or t2 != 0:
            print("programming frames F9 to 0B: {}    programming frames 0B to F9: {} \n".format(t1, t2))

        '''
        if xx[3] == '0x18EAFFFE':
            print("REQUEST ", xx)

        if xx[3] == "0x18EEFFF9":
            print("ADDRESS CLAIMED RESPONSE ", xx)

        if xx[3] == "0x18DFFFF9":
            print("ADDRESS CLAIMED RESPONSE ", xx)
        '''
        time[j] = datetime.strptime(xx[0], format)
        if j == 0:
            date_time[j] = temporary
        else:
            date_time[j] = time[j] - temporary

        temporary = datetime.strptime(xx[0], format)

        k = 0
        l = 0
        p1 = 0
        p2 = 0
        q1 = 0
        q2 = 0
        r1 = 0
        r2 = 0
        s1 = 0
        s2 = 0
        t1 = 0
        t2 = 0
        j += 1


    if xx[3] == '0x18DA3DF9':

        k += 1

    if xx[3] == '0x18DAF93D':

        l +=1

    if xx[3] == '0x18DAF94D':

        p1 += 1

    if xx[3] == '0x18DA4DF9':

        p2 += 1

    if xx[3] == '0x18DAF927':

        q1 += 1

    if xx[3] == '0x18DA27F9':

        q2 += 1

    if xx[3] == '0x18DAF925':

        r1 += 1

    if xx[3] == '0x18DA25F9':

        r2 += 1

    if xx[3] == '0x18DAF900':

        s1 += 1

    if xx[3] == '0x18DA00F9':

        s2 += 1

    if xx[3] == '0x18DAF90B':

        t1 += 1

    if xx[3] == '0x18DAOBF9':

        t2 += 1
    '''
    if xx[3] == "0x18DAF925" or xx[3] == '0x18DA25F9':

    if xx[3] == '0x1CFDD127':
        print(xx)

        n += 1
    '''


    if pgn not in pgn_dict:

        pgn_dict[pgn] = 1
    else:
        pgn_dict[pgn] = pgn_dict[pgn] + 1

print(pgn_dict)
print(n)

post_rqst_time[:] = (value for value in post_rqst_time if value != 0)
for i in range(len(post_rqst_time)):
    print(post_rqst_time[i])

'''
date_time[:] = (value for value in date_time if value != 0)

for x in range(len(date_time)):
    print(date_time[x])
'''


'''
main_dir= 'D:/Meus documentos/Desktop/mAP comparison files/KITTI_MOD_fixed/testing/'
imgs_dir= main_dir+'images/'
boxes_dir= main_dir+'boxes/'
txt_count = 175

for f in sorted(os.listdir(imgs_dir)):



    img = cv2.imread(imgs_dir+f)
    box_file= open(boxes_dir+f.split('.')[0]+'.txt', 'r')

    boxes, labels, classes= parse_file(box_file)

    with open("C:/Python/YOLO_3D_HART/text_to_eval/groundtruths/0000{}.txt".format(txt_count), "w") as text_file:

        for i in range(len(boxes)):

                if classes[i] == "Car":
                    classes[i] = "car"

                if classes[i] == "Van":
                    classes[i] = "truck"


                print("{} {} {} {} {}".format(classes[i], boxes[i][1], boxes[i][0], boxes[i][3] , boxes[i][2]), file=text_file)

    txt_count += 1

    boxes_img= draw_boxes(boxes, labels, classes, img)

    #cv2.imshow('IMG ', img)
    cv2.imshow('Boxes IMG ', boxes_img)
    print(f)
    cv2.waitKey()
'''
