import imageio
from moviepy.editor import *

# Replace 'input.avi' with the name of your input file
clip = VideoFileClip("E:/Study/jason/CIITR-LiDAR-main/New Dev/Track_SGH/output/demo.avi")

# Replace 'output.gif' with the name of your output file
clip.write_gif("E:/Study/jason/CIITR-LiDAR-main/New Dev/Track_SGH/output/output.gif")




