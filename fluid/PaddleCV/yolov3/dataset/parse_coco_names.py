from pycocotools.coco import COCO

coco = COCO("./coco/annotations/instances_val2014.json")

f = open("coco.names", 'w')
cats = coco.loadCats(coco.getCatIds())
for c in cats:
    print c
    label_name = c['name']
    label_id = int(c['id'])

    f.write("{} {}\n".format(label_id, label_name))
f.close()
