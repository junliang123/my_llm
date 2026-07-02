import xml.etree.ElementTree as ET

def parse_xml(xml_path, class_to_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.findtext("filename")
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")

        xmin = int(bndbox.findtext("xmin"))
        ymin = int(bndbox.findtext("ymin"))
        xmax = int(bndbox.findtext("xmax"))
        ymax = int(bndbox.findtext("ymax"))

        x_center = (xmin + xmax)/(2*width)
        y_center = (ymin + ymax)/(2*height)
        box_w = (xmax - xmin)/width
        box_h = (ymax - ymin)/height

        labels.append((class_to_id[name], x_center, y_center, box_w, box_h))

    return filename, labels