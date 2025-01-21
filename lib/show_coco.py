import os
import sys
import numpy as np
from lib import visualize
from lib import utils

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

############################################################
#  Configurations
############################################################
class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def __init__(self, dataset_dir, annotations, images_path, class_ids=None,
                  class_map=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        super(CocoDataset,self).__init__()
        self.coco = COCO("{}/{}".format(dataset_dir, annotations))
        image_dir = "{}/{}".format(dataset_dir, images_path)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(self.coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_Ids = []
            for id in class_ids:
                image_Ids.extend(list(self.coco.getImgIds(catIds=[id])))
            # Remove duplicates
            self.image_ids = list(set(image_Ids))
        else:
            # All images
            self.image_ids = list(self.coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, self.coco.loadCats(i)[0]["name"])

        # Add images
        for i in self.image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, self.coco.imgs[i]['file_name']),
                width=self.coco.imgs[i]["width"],
                height=self.coco.imgs[i]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_normalmask(self, image_id, img_size, class_channels = True):
        """Load semantic segmenation masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, classes].

        Returns:
        masks: A bool array of shape [height, width, classes] with
            one mask per class.
        """
        image_info = self.image_info[image_id]

        if class_channels:
            mask = np.zeros((img_size[0],img_size[1], self.num_classes))
        else:
            mask = np.zeros((img_size[0],img_size[1]))

        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:
            # className = self.class_info[annotation['category_id']]['name']
            # pixel_value = self.class_names.index(className)
            ann_mask = self.coco.annToMask(annotation)
            ann_mask = utils.resize(ann_mask, img_size)
            ann_mask[ann_mask!=0] = 1 #convert nonzero values to 1
            pixel_value = annotation['category_id']
            # pixel_value = self.map_source_class_id(f'{self.class_info[]['source']}.{}')
            if class_channels:
                mask[:,:,pixel_value] = np.maximum(ann_mask ,mask[:,:,pixel_value]) 
            else:
                mask = np.maximum(ann_mask*pixel_value, mask)
        return mask
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


if __name__ == '__main__':
    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0

    config = InferenceConfig()

    # Validation dataset
    dataset_dir = './NWPU VHR-10_dataset_coco'
    annotations = 'instances_val2017.json'
    images_path = 'positive image set'

    dataset = CocoDataset(dataset_dir, annotations, images_path)
    dataset.prepare()

    # print(dataset.class_info)
    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # Load random image and mask.
    image_id = np.random.choice(dataset.image_ids, 1)[0]
    image = dataset.load_image(image_id)
    mask_1, class_ids = dataset.load_mask(image_id)
    # print(mask_1)
    # mask = dataset.load_normalmask(image_id)
    # print(image.shape, mask.shape)
    # print(np.unique(mask))
    # # Compute Bounding box
    bbox = utils.extract_bboxes(mask_1)

    # # Display image and instances
    visualize.display_instances(image, bbox, mask_1, class_ids, dataset.class_names)




