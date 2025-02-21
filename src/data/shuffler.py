import random


class Shuffler:
    """Shuffles in-memory frames and annotations to remove temporal bias."""

    def __init__(self, seed=None):
        self.seed = seed
        if self.seed:
            random.seed(self.seed)

    def shuffle(self, data):
        """
        Shuffles frames and annotations while preserving their links.

        :param data: List of dictionaries with {"frame": tf.Tensor, "annotation": dict}
        :return: Shuffled list
        """
        random.shuffle(data)
        return data  # Returns only shuffled data, nothing else
