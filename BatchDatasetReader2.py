"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import cv2


class BatchDataset2:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, numAnnotationClasses, image_options={}, fromFrameId = 0, uptoFrameId = None):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options

        self.fromFrameId = 0
        self.uptoFrameId = uptoFrameId

        self.numAnnotationClasses = numAnnotationClasses
        self._read_images()

    def _read_images(self):
        self.images = np.array([self._transform(filename['image']) for filename in self.files if self.uptoFrameId is None or self.fromFrameId <= filename['frameId'] < self.uptoFrameId])
        self.annotations = np.array(
            [self._transform(filename['annotation'], annotations=True) for filename in self.files if self.uptoFrameId is None or self.fromFrameId <= filename['frameId'] < self.uptoFrameId])
        print (self.images.shape)
        print (self.annotations.shape)

    def _transform(self, filename, annotations=False):
        image = cv2.imread(filename)

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            image = cv2.resize(image,(resize_size, resize_size), cv2.INTER_NEAREST)

        if annotations:
            mask = np.zeros((image.shape[0], image.shape[1], self.numAnnotationClasses), dtype=np.float32)
            mask[:, :, 0] = 1.0

            color_classes = [(255, 255, 255), ]
            for i, color in enumerate(color_classes):
                isColor = np.logical_and(np.logical_and(image[:, :, 0] == color[0], image[:, :, 1] == color[1]), image[:, :, 2] == color[2])
                isColorI = np.argwhere(isColor)

                value = np.zeros((self.numAnnotationClasses), dtype=np.float32)
                value[i + 1] = 1.0

                mask[isColorI[:, 0], isColorI[:, 1], :] = value

            image = mask

        return np.array(image)

    def changeFrameIdRange(self, fromFrameId, uptoFrameId):
        self.fromFrameId = fromFrameId
        self.uptoFrameId = uptoFrameId
        self._read_images()

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]