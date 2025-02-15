import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os

def flip(address,saveadd):
    img = Image.open(address)
    img2 = ImageOps.mirror(img)
    img2.save(saveadd + "_f.png")

def bright(address,saveadd):
    im = Image.open(address)
    enhancer = ImageEnhance.Brightness(im)
    factor = 1.5
    im_output = enhancer.enhance(factor)
    im_output.save(saveadd + "_b1.png")

    factor = 0.5 #gives original image
    im_output = enhancer.enhance(factor)
    im_output.save(saveadd + "_b2.png")

def blur(address,saveadd):
    OriImage = Image.open(address)

    blurImage = OriImage.filter(ImageFilter.BLUR)
    blurImage.save(saveadd + "_bl.png")

def main():
    arr = sorted(os.listdir("cropped/Engaged"))
    arr2 = sorted(os.listdir("cropped/Unengaged"))
    count = 0
    for i in range(0, len(arr)):
        path = "cropped/Engaged/" + arr[i]
        flip(path, "new/Engaged/" + str(count))
        bright(path, "new/Engaged/" + str(count))
        blur(path, "new/Engaged/" + str(count))
        os.rename(path, r'new/Engaged/' + str(count) + ".png")
        count += 1
    count2 = 0
    for j in range(0,len(arr2)):
        path = "cropped/Unengaged/" + arr2[j]
        flip(path, "new/Unengaged/" + str(count2))
        bright(path, "new/Unengaged/" + str(count2))
        blur(path, "new/Unengaged/" + str(count2))
        os.rename(path, r'new/Unengaged/' + str(count2) + ".png")
        count2 += 1

#main()
def normalize(folder):
    arr = sorted(os.listdir(folder + "Engaged"))
    arr2 = sorted(os.listdir(folder + "Unengaged"))
    mean1 = [0,0,0,0]
    std1 = [0,0,0,0]
    for i in range(0,len(arr)):
        if arr[i] != ".DS_Store":
            path = folder + "Engaged/" + arr[i]
            image = Image.open(path)
            data = np.asarray(image)
            for a in range(0,4):
                mean1[a] += np.mean(data[:,:,a])
                std1[a] += np.std(data[:,:,a])
    for j in range(0,len(arr2)):
        if arr[j] != ".DS_Store":
            path = folder + "Unengaged/" + arr[j]
            image = Image.open(path)
            data = np.asarray(image)
            for b in range(0,4):
                mean1[b] += np.mean(data[:,:,b])
                std1[b] += np.std(data[:,:,b])
    for i in range(0,4):
        mean1[i] = mean1[i] / (len(arr) + len(arr2))
        std1[i] = std1[i] / (len(arr) + len(arr2))
    print(mean1)
    print(std1)

normalize("new/")
