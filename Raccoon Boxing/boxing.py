def draw_boxes(image_name):
    import cv2
    
    selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread('images/{}'.format(image_name))
    for index, row in selected_value.iterrows():
        img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 3)
    return img