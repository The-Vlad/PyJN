def draw_mask(img, mask, sr=1., sens=0.9):
    from PIL import ImageFilter
    from keras.preprocessing import image
    
    img = image.array_to_img(img)
    mask = image.array_to_img(mask)
    imgSize = max(img.size)
    maskSize = min(mask.size)
    mask = mask.resize(img.size).filter(ImageFilter.GaussianBlur(radius = imgSize/maskSize*sr if maskSize<imgSize else 0))
    img = image.img_to_array(img)
    mask = image.img_to_array(mask)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if mask[x][y][0] >= sens:
                img[x][y][1] += 50
    return image.array_to_img(img)