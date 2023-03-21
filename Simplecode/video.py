import cv2
import os
path = r'E:\\Study\\jason\\CIITR-LiDAR-main\\New Dev\\Track_SGH\\output\\image3/'
out_path= r'E:\Study\\jason\\CIITR-LiDAR-main\\New Dev\\Track_SGH\\output/'
out_video_name ='demo.avi'
out_video_full_path= out_path+out_video_name
img=[]
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0])) #将每个文件名将句号 (.) 前的字符串转化为数字，然后以数字为key来进行排序。

for i in path_list:
    i = path+i
    img.append(i)
cv2_fourcc= cv2.VideoWriter_fourcc(*'MJPG')
frame=cv2.imread(img[0])
size =list(frame.shape)
# print(size)
del size[2]
size.reverse()
video = cv2.VideoWriter(out_video_full_path, cv2_fourcc,15,size) #output video name

for i in range(len(img)):
    video.write(cv2.imread(img[i]))

video.release()
# import cv2
# import os
# from os import listdir

# image_folder =  'E:\\Study\\jason\\CIITR-LiDAR-main\\New Dev\\Track_SGH\\output\\image3/'
# out_path= 'E:\Study\\jason\\CIITR-LiDAR-main\\New Dev\\Track_SGH\\output/'
# video_name = 'video.avi'
# out_video_full_path= out_path+video_name

# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
# for i in os.listdir(image_folder):
#     i=i.png



# # video = cv2.VideoWriter(out_video_full_path, 0, 5, (width,height))

# # for image in images:
# #     video.write(cv2.imread(os.path.join(image_folder, image)))

# # cv2.destroyAllWindows()
# # video.release()
# #path_list.sort(key=lambda x:int(x[:-4]))