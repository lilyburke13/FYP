#!/bin/bash

## For each image in the positive samples folder
## Execute the OpenCV create samples function.

## Change this number according to the amount of positive samples available
NUM=70

for filename in ../positive_samples/*.jpg; do
    opencv_createsamples -img ../positive_samples/$filename -bg ../negatives.txt -num $NUM -info info.txt
done
