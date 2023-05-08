# %%
import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import functions_train as ft

# %%

_DEFAULT_OUTPUT_DIR = './output'

_DEFAULT_DURATION = 0.5  # seconds
_DEFAULT_SAMPLE_RATE = 44100

_DEFAULT_TEST_SIZE = 0.1
_DEFAULT_VAL_SIZE = 0.1

_DEFAULT_NUM_SHARDS_TRAIN = 28
_DEFAULT_NUM_SHARDS_TEST = 10
_DEFAULT_NUM_SHARDS_VAL = 0


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


class TFRecordsConverter:
    """Convert audio to TFRecords."""

    def __init__(self, output_dir, n_shards_train, n_shards_test,
                 n_shards_val, duration, sample_rate, test_size, val_size):
        self.output_dir = output_dir
        self.n_shards_train = n_shards_train
        self.n_shards_test = n_shards_test
        self.n_shards_val = n_shards_val
        self.duration = duration
        self.sample_rate = sample_rate

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # df = pd.read_csv(meta, index_col=0)

        path, labels = ft.extract_input_target()
        # Shuffle data by "sampling" the entire data-frameù
        labels = labels+labels
        self.df = list(zip(path, labels))
        random.shuffle(self.df)
        n_samples = len(self.df)
        self.n_test = math.ceil(n_samples * test_size)
        self.n_val = math.ceil(n_samples * val_size)
        self.n_train = n_samples - self.n_test - self.n_val

    def _get_shard_path(self, split, shard_id, shard_size):
        return os.path.join(self.output_dir,
                            '{}-{:03d}-{}.tfrecord'.format(split, shard_id,
                                                           shard_size))

    def _write_tfrecord_file(self, shard_path, indices):
        """Write TFRecord file."""
        with tf.io.TFRecordWriter(shard_path) as out:
            for index in indices:
                try:
                    file_path = self.df[index][0]
                    label = self.df[index][1]

                    raw_audio = tf.io.read_file(file_path)
                    audio, _ = tf.audio.decode_wav(raw_audio)

                    # Example is a flexible message type that contains key-value
                    # pairs, where each key maps to a Feature message. Here, each
                    # Example contains two features: A FloatList for the decoded
                    # audio data and an Int64List containing the corresponding
                    # label's index.
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'audio': _float_feature(audio.numpy().flatten().tolist()),
                        'label': _float_feature(label)}))
                    out.write(example.SerializeToString())
                except ValueError as e:
                    print(
                        f"Skipping record {index} due to missing data: {str(e)}")

    def convert(self):
        """Convert to TFRecords.
        Partition data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        """
        splits = ('train', 'test')
        split_sizes = (self.n_train, self.n_test)
        split_n_shards = (self.n_shards_train, self.n_shards_test)

        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            print('Converting {} set into TFRecord shards...'.format(split))
            shard_size = math.ceil(size / n_shards)
            cumulative_size = offset + size
            for shard_id in range(1, n_shards + 1):
                step_size = min(shard_size, cumulative_size - offset)
                shard_path = self._get_shard_path(split, shard_id, step_size)
                # Generate a subset of indices to select only a subset of
                # audio-files/labels for the current shard.
                file_indices = np.arange(offset, offset + step_size)
                self._write_tfrecord_file(shard_path, file_indices)
                offset += step_size

        print('Number of training examples: {}'.format(self.n_train))
        print('Number of testing examples: {}'.format(self.n_test))
        print('Number of validation examples: {}'.format(self.n_val))
        print('TFRecord files saved to {}'.format(self.output_dir))


def main():
    converter = TFRecordsConverter(output_dir=_DEFAULT_OUTPUT_DIR,
                                   n_shards_train=_DEFAULT_NUM_SHARDS_TRAIN,
                                   n_shards_test=_DEFAULT_NUM_SHARDS_TEST,
                                   n_shards_val=_DEFAULT_NUM_SHARDS_VAL,
                                   duration=_DEFAULT_DURATION,
                                   sample_rate=_DEFAULT_SAMPLE_RATE,
                                   test_size=_DEFAULT_TEST_SIZE,
                                   val_size=_DEFAULT_VAL_SIZE)
    converter.convert()


# %%
if __name__ == '__main__':
    main()
# %%

# %%

raw_dataset = tf.data.TFRecordDataset('./output/train-001-17856.tfrecord')
feature_description = {
    'audio': tf.io.FixedLenFeature([22050], tf.float32),
    'label': tf.io.FixedLenFeature([2], tf.float32),
}


def _parse_example_proto(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


parsed_audio_dataset = raw_dataset.map(_parse_example_proto)

audio_raw = []
label = []
for example in parsed_audio_dataset:
    audio_raw.append(example['audio'].numpy())
    label.append(example['label'].numpy())
# %%
