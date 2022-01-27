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

# import _setup_returnn_env  # noqa
from returnn.log import log
import returnn.__main__ as rnn
import returnn.datasets.hdf as hdf_dataset_mod
from returnn.datasets import Dataset, init_dataset
from returnn.config import Config
from returnn.tf.util.data import Data

tf.compat.v1.enable_eager_execution()

def hdf_dataset_init(dataset, file_name):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  # return hdf_dataset_mod.HDFDatasetWriter(filename=file_name)
  return hdf_dataset_mod.SimpleHDFWriter(
    filename=file_name, dim=dataset.get_data_dim("classes") + 1, ndim=1)

def hdf_dump_from_dataset(dataset, hdf_dataset, parser_args):
  """
  :param Dataset dataset: could be any dataset implemented as child of Dataset
  :param hdf_dataset_mod.HDFDatasetWriter hdf_dataset:
  :param parser_args: argparse object from main()
  """
  # hdf_dataset.dump_from_dataset(
  #   dataset=dataset,
  #   epoch=parser_args.epoch, start_seq=parser_args.start_seq, end_seq=parser_args.end_seq,
  #   use_progress_bar=True)

  seq_idx = parser_args.start_seq
  end_idx = parser_args.end_seq
  if end_idx < 0:
    end_idx = float("inf")
  dataset.init_seq_order(parser_args.epoch)
  target_dim = dataset.get_data_dim("classes")
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_idx:
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "classes")
    print("ORIGINAL: ", data)
    blank_mask = [True if i == j else False for i, j in zip(data[:-1], data[1:])] + [False]
    data[blank_mask] = target_dim
    print("MODIFIED: ", data)
    seq_len = dataset.get_seq_length(seq_idx)["classes"]
    tag = dataset.get_tag(seq_idx)
    data_obj = Data(name="alignment", sparse=True, dim=89, size_placeholder={0: tf.constant([seq_len])}, placeholder=tf.constant(np.expand_dims(data, axis=0), dtype="int32"))
    new_data = data_obj.copy_as_batch_spatial_major().placeholder.numpy()

    extra = {}
    sizes = tf.convert_to_tensor([size for (i, size) in sorted(data_obj.size_placeholder.items())])
    seq_lens = {i: size.numpy() for (i, size) in zip(sorted(data_obj.size_placeholder.keys()), sizes)}
    ndim_without_features = data_obj.ndim - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    print("New Data: ", new_data)

    hdf_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)
    seq_idx += 1


def hdf_close(hdf_dataset):
  """
  :param HDFDataset.HDFDatasetWriter hdf_dataset: to close
  """
  hdf_dataset.close()


def init(rasr_config_path, time_red):
  """
  :param str config_filename: global config for CRNN
  :param list[str] cmd_line_opts: options for init_config method
  :param str dataset_config_str: dataset via init_dataset_via_str()
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  log.initialize(verbosity=[5])
  print("Returnn hdf_dump starting up.", file=log.v3)
  rnn.init_faulthandler()

  sprint_args = [
    "--config=%s" % rasr_config_path,
    "--*.LOGFILE=nn-trainer.train.log",
    "--*.TASK=1", "--*.corpus.segment-order-shuffle=true",
    "--*.reduce-alignment-factor=%d" % time_red
  ]

  dataset_dict = {
    'class': 'ExternSprintDataset',
    # 'sprintConfigStr': '--config=%s --*.LOGFILE=nn-trainer.train.log --*.TASK=1 --*.corpus.segment-order-shuffle=true' % rasr_config_path,
    'sprintConfigStr': sprint_args,
    'sprintTrainerExecPath': '/u/zhou/rasr-dev/arch/linux-x86_64-standard-label_sync_decoding/nn-trainer.linux-x86_64-standard-label_sync_decoding'}
  dataset = init_dataset(dataset_dict)
  print("Source dataset:", dataset.len_info(), file=log.v3)
  return dataset


def _is_crnn_config(filename):
  """
  :param str filename:
  :rtype: bool
  """
  if filename.endswith(".gz"):
    return False
  if filename.endswith(".config"):
    return True
  # noinspection PyBroadException
  try:
    config = Config()
    config.load_file(filename)
    return True
  except Exception:
    pass
  return False


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

  args = parser.parse_args(argv[1:])



  dataset = init(rasr_config_path=args.rasr_config, time_red=args.time_red)
  hdf_dataset = hdf_dataset_init(dataset, args.hdf_filename)
  hdf_dump_from_dataset(dataset, hdf_dataset, args)
  hdf_close(hdf_dataset)

  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
