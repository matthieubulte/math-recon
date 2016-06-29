from shapes.image import *

class DataSet:
    def __init__(self, directory, batch_size=64, test_part=0.1):
        images, labels = DataSet.__load_data(directory)
        images, labels = DataSet.__shuffle(images, labels)

        split_index = int(test_part * len(images))

        self.test_images = images[: split_index]
        self.test_labels = labels[: split_index]

        self.train_images = images[split_index :]
        self.train_labels = labels[split_index :]

        self.batch_size = batch_size
        self.cursor_index = 0
        self.training_dataset_length = len(self.train_images)

    def next_batch(self):
        if self.cursor_index + self.batch_size > self.training_dataset_length:
            self.cursor_index = 0
            self.train_images, self.train_labels = DataSet.__shuffle(self.train_images, self.train_labels)

        start = self.cursor_index
        self.cursor_index += self.batch_size

        return self.train_images[start:self.cursor_index], self.train_labels[start:self.cursor_index]

    @staticmethod
    def __load_data(directory):
        file_template = directory + "img%03d-%03d.png"

        # read dataset
        images = []
        labels = []
        for i in xrange(1, 63):
            for j in xrange(1, 55 * 6):
                f = file_template % (i, j)

                label_index = i - 1

                # small letters should be classified the same as large letters
                if i >= 37:
                    label_index -= 26

                label = np.zeros(36)
                label[label_index] = 1

                images.append(Image.from_file(f).flatten())
                labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    @staticmethod
    def __shuffle(images, labels):
        perm = np.arange(len(images))
        np.random.shuffle(perm)
        return images[perm], labels[perm]
