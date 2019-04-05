mport os
import shutil

## Count the number of images in each file
## Delete the file if the count is less than 15 images
for filename in os.listdir('.'):
    size = sum([len(files) for r, d, files in os.walk(filename)])
    if size < 15:
        shutil.rmtree(filename)
