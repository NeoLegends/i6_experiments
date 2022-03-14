#!/usr/bin/env python

"""
For debugging, go through some dataset, forward it through the net, and output the layer activations on stdout.
"""

from __future__ import print_function

import sys

# import returnn.__main__ as rnn
# from returnn.log import log
import argparse
# from returnn.util.basic import pretty_print
from subprocess import check_output
import tensorflow as tf
import numpy as np


def dump(ref_dataset, search_dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  output_dict = {}
  layer = rnn.engine.network.get_layer("output/output_log_prob")
  output_dict["%s-out" % "output_log_prob"] = layer.output.get_placeholder_as_batch_major()

  seq_idx = 0
  num_search_errors = 0

  while ref_dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= float("inf"):
    ref_out = rnn.engine.run_single(dataset=ref_dataset, seq_idx=seq_idx, output_dict=output_dict)
    search_out = rnn.engine.run_single(dataset=search_dataset, seq_idx=seq_idx, output_dict=output_dict)
    # get probs for ref and search alignment
    ref_output_prob = ref_out["output_log_prob-out"][0]  # [T, V + 1]
    search_output_prob = search_out["output_log_prob-out"][0]  # [T, V + 1]
    # get ground truth alignment and search alignment
    ref_dataset.load_seqs(seq_idx, seq_idx + 1)
    search_dataset.load_seqs(seq_idx, seq_idx + 1)
    ref_align = ref_dataset.get_data(seq_idx, "alignment")
    search_align = search_dataset.get_data(seq_idx, "alignment")
    ref_log_probs_seq = np.take_along_axis(ref_output_prob, ref_align[:, None], axis=-1)
    search_log_probs_seq = np.take_along_axis(search_output_prob, search_align[:, None], axis=-1)
    ref_log_score = np.sum(ref_log_probs_seq)
    search_log_score = np.sum(search_log_probs_seq)

    print("REF FEED LOG SCORE: ", ref_log_score)
    print("SEARCH FEED LOG SCORE: ", search_log_score)
    # print("PROB SEQ: ", probs_seq)
    print("REF ALIGN: ", ref_align)
    print("SEARCH ALIGN: ", search_align)

    if search_log_score < ref_log_score:
      num_search_errors += 1

    seq_idx += 1

  with open("search_errors", "w+") as f:
    f.write(str(num_search_errors / seq_idx))


def net_dict_add_losses(net_dict):
  net_dict["output"]["unit"]["output_log_prob"]["is_output_layer"] = True

  return net_dict

def init(config_filename, corpus_file, segment_file, feature_cache, ref_align, search_align):
  """
  :param str config_filename:
  :param list[str] command_line_options:
  """
  rnn.init(
    config_filename=config_filename,
    config_updates={"log": None},
    extra_greeting="RETURNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config, train_data=rnn.train_data)
  rnn.engine.init_network_from_config(net_dict_post_proc=net_dict_add_losses)

  def add_cf(file_path):
    return "'`cf " + file_path + "`'"

  args = ["--config=/u/schmitt/experiments/transducer/config/rasr-configs/extern_sprint_dataset.config",
          "--*.corpus.file=" + corpus_file,
          "--*.corpus.segments.file=" + segment_file if segment_file else "", "--*.log-channel.file=sprint-log",
          "--*.feature-cache-path=" + add_cf(feature_cache) if check_output(["hostname"]).strip().decode("utf8") != "cluster-cn-211" else "--*.feature-cache-path=" + feature_cache,
          "--*.window-size=1"]

  d = {
    "class": "ExternSprintDataset",
    "sprintTrainerExecPath": "/u/zhou/rasr-dev/arch/linux-x86_64-standard-label_sync_decoding/nn-trainer.linux-x86_64-standard-label_sync_decoding",
    "sprintConfigStr": args, "suppress_load_seqs_print": True,  # less verbose
    "input_stddev": 3.}

  ref_align_opts = {
    "class": "HDFDataset", "files": [ref_align], "use_cache_manager": True,
    "seq_list_filter_file": segment_file}

  search_align_opts = {
    "class": "HDFDataset", "files": [search_align], "use_cache_manager": True, "seq_list_filter_file": segment_file}

  d_ref = {
    "class": "MetaDataset", "datasets": {"sprint": d, "align": ref_align_opts}, "data_map": {
      "data": ("sprint", "data"),
      "alignment": ("align", "data"),
    }, "seq_order_control_dataset": "align",
  }

  d_search = {
    "class": "MetaDataset", "datasets": {"sprint": d, "align": search_align_opts}, "data_map": {
      "data": ("sprint", "data"),
      "alignment": ("align", "data"),
    }, "seq_order_control_dataset": "align",
  }

  ref_dataset = rnn.init_dataset(d_ref)
  search_dataset = rnn.init_dataset(d_search)

  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()

  return ref_dataset, search_dataset


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('returnn_config')
  arg_parser.add_argument('--corpus_file')
  arg_parser.add_argument('--segment_file')
  arg_parser.add_argument('--feature_cache')
  arg_parser.add_argument('--ref_align')
  arg_parser.add_argument('--search_align')
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  ref_dataset, search_dataset = init(
    config_filename=args.returnn_config, corpus_file=args.corpus_file, segment_file=args.segment_file,
    ref_align=args.ref_align, search_align=args.search_align, feature_cache=args.feature_cache)
  dump(ref_dataset, search_dataset, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
