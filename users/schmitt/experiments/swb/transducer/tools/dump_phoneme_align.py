#!/usr/bin/env python3

"""
Creates a HDF file, which can be read by :class:`HDFDataset`.
The input is any other dataset (:class:`Dataset`).
"""

from __future__ import print_function

import sys
import argparse
import numpy as np
import tensorflow as tf


def hdf_dataset_init(dataset, file_name):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename=file_name, dim=dataset.get_data_dim("classes") + 2, ndim=1)


def hdf_dump_from_dataset(dataset, hdf_dataset, parser_args):
  """
  :param Dataset dataset: could be any dataset implemented as child of Dataset
  :param hdf_dataset_mod.HDFDatasetWriter hdf_dataset:
  :param parser_args: argparse object from main()
  """
  seq_idx = parser_args.start_seq
  end_idx = parser_args.end_seq
  if end_idx < 0:
    end_idx = float("inf")
  dataset.init_seq_order(parser_args.epoch)
  target_dim = dataset.get_data_dim("classes")
  blank_idx = target_dim + 1  # idx target_dim is reserved for SOS token
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_idx:
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "classes")
    blank_mask = [True if i == j else False for i, j in zip(data[:-1], data[1:])] + [False]
    data[blank_mask] = target_dim
    seq_len = dataset.get_seq_length(seq_idx)["classes"]
    tag = dataset.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(data, axis=0), dtype="int32")

    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1 #- (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    hdf_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)
    seq_idx += 1


def hdf_close(hdf_dataset):
  """
  :param HDFDataset.HDFDatasetWriter hdf_dataset: to close
  """
  hdf_dataset.close()


def init(rasr_config_path, time_red, data_key):
  """
  :param str config_filename: global config for CRNN
  :param list[str] cmd_line_opts: options for init_config method
  :param str dataset_config_str: dataset via init_dataset_via_str()
  """
  from returnn.log import log
  from returnn.__main__ import init_better_exchook, init_thread_join_hack, init_faulthandler
  init_better_exchook()
  init_thread_join_hack()
  log.initialize(verbosity=[5])
  print("Returnn hdf_dump starting up.", file=log.v3)
  init_faulthandler()

  estimated_num_seqs = {"train": 227047, "cv": 3000, "devtrain": 3000}
  epoch_split = 1

  sprint_args = [
    "--config=%s" % rasr_config_path,
    "--*.LOGFILE=nn-trainer.train.log",
    "--*.TASK=1", "--*.corpus.segment-order-shuffle=true",
    "--*.reduce-alignment-factor=%d" % time_red, "--*.corpus.segment-order-shuffle=true",
    "--*.segment-order-sort-by-time-length=true",
    "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data_key, -1),
  ]

  dataset_dict = {
    'class': 'ExternSprintDataset',
    "reduce_target_factor": time_red,
    'sprintConfigStr': sprint_args,
    'sprintTrainerExecPath': '/u/zhou/rasr-dev/arch/linux-x86_64-standard-label_sync_decoding/nn-trainer.linux-x86_64-standard-label_sync_decoding',
    "partition_epoch": epoch_split,
    "estimated_num_seqs": (estimated_num_seqs[data_key] // epoch_split) if data_key in estimated_num_seqs else None, }
  dataset = rnn.datasets.init_dataset(dataset_dict)
  print("Source dataset:", dataset.len_info(), file=log.v3)
  return dataset


def main(argv):
  """
  Main entry.
  """
  parser = argparse.ArgumentParser(description="Dump dataset or subset of dataset into external HDF dataset")
  parser.add_argument('rasr_config', type=str,
                      help="Config file for RETURNN, or directly the dataset init string")
  parser.add_argument('hdf_filename', type=str, help="File name of the HDF dataset, which will be created")
  parser.add_argument('--start_seq', type=int, default=0, help="Start sequence index of the dataset to dump")
  parser.add_argument('--end_seq', type=int, default=float("inf"), help="End sequence index of the dataset to dump")
  parser.add_argument('--epoch', type=int, default=1, help="Optional start epoch for initialization")
  parser.add_argument('--time_red', type=int, default=1, help="Time-downsampling factor")
  parser.add_argument('--returnn_root', help="Returnn root to use for imports")
  parser.add_argument('--data_key')

  args = parser.parse_args(argv[1:])

  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn as rnn
  import returnn.__main__ as rnn_main
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  dataset = init(rasr_config_path=args.rasr_config, time_red=args.time_red, data_key=args.data_key)
  hdf_dataset = hdf_dataset_init(dataset, args.hdf_filename)
  hdf_dump_from_dataset(dataset, hdf_dataset, args)
  hdf_close(hdf_dataset)

  rnn_main.finalize()


if __name__ == '__main__':
  main(sys.argv)
