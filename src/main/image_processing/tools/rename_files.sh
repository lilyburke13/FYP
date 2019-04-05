#!/bin/bash

## Initiate counter
counter=0

## Get the name of the folder
name=${PWD##*/}

## For each image in the folder, rename it with
## the folder name and ccurrent counter value
for file in *; do
    [[ -f $file ]] && mv -i "$file" ${name}_$((counter+1)).jpg && ((counter++))
done
