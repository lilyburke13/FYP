import os
from PIL import Image
import sys


## Define image size
size = 250, 300
## Initiate counter
count = 1

## For every image in the tmp folder
## Resize it to the size defined above
for image_name in os.listdir("tmp"):
    image = Image.open("tmp/"+image_name)
    image.thumbnail(size, Image.ANTIALIAS)
    image.save("positive_samples/"+ image_name + "_" +str(count)+".jpg","JPEG")
    count += 1
