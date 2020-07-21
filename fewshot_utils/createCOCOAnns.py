import numpy as np
import os
import glob
from PIL import Image
# from pycocotools import mask
import xml.etree.ElementTree as ET
import pylab as plt
from matplotlib.path import Path



def drawSegImg(img_file, xml_file):
    image_size, labelMap = readLabelMap(xml_file)
    image = plt.imread(img_file)
    save_file = img_file.replace('Images', 'Save')

    height, width = image_size[0], image_size[1]
    # print(image_size)

    poly_path=Path(labelMap)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((y.reshape(-1,1), x.reshape(-1, 1))) # coors.shape is (4000000,2)

    mask = poly_path.contains_points(coors)
    mask=mask.reshape(height, width)
    image = image*mask[:,:,None]
    image = Image.fromarray(image)
    # crop boxes
    labelMap = np.array(labelMap)
    x1 = labelMap[:,0].min()
    y1 = labelMap[:,1].min()
    x2 = labelMap[:,0].max()
    y2 = labelMap[:,1].max()
    image.crop((x1, y1, x2, y2))
    image.save(save_file)

def readLabelMap(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # assert root[0].text=='annotation', root.text

    annotation = root
    labelMap = []
    polygon = None
    image_w = None
    image_h = None

    for module in annotation:
        if module.tag=='imagesize':
            image_w = int(module[0].text)
            image_h = int(module[1].text)
        if module.tag=='object':
            for mini_module in module:
                if mini_module.tag=='polygon':
                    polygon=mini_module

    assert polygon!=None     
    for pt in polygon:
        if pt.tag=='pt':
            x=int(pt[0].text)
            y=int(pt[1].text)
            labelMap.append([x,y])
    return (image_w, image_h), labelMap


if __name__ == "__main__":
    dir_name = 'voc2007_test_coco'
    xml_dir = os.path.join(dir_name, 'Annotations')
    img_dir = os.path.join(dir_name, 'Images')
    save_dir = os.path.join(dir_name, 'Save')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    xml_files.sort()
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_files.sort()
    assert len(xml_files)==len(img_files), (xml_files, img_files)

    for i in range(len(xml_files)):
        drawSegImg(img_files[i], xml_files[i])


