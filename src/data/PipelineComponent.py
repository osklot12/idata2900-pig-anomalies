import tensorflow as tf
from abc import ABC, abstractmethod

class PipelineComponent(ABC):
    """
    Abstract base class for TensorFlow pipeline components.
    Ensures all modules implement a `process` method and work with `tf.data.Dataset`.
    """

    def __call__(self, data: tf.Tensor) -> tf.Tensor:
        """Allows objects of this class to be used as functions in `dataset.map()`"""
        return self.process(data)


    @abstractmethod
    def process(self, data: tf.Tensor) -> tf.Tensor:
        """Processes the input data and returns the transformed output."""
        pass


    def to_tf_function(self):
        """Converts the process method into a TensorFlow graph function for efficiency."""
        self.process = tf.function(self.process, autograph=True)


    def debug(self, sample_data):
        """Helper method to test the transformation on a sample."""
        transformed_data = self.process(tf.convert_to_tensor(sample_data))
        print("Transformed Data: ", transformed_data)