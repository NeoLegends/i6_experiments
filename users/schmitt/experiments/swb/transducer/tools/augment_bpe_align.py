import argparse
import sys
import numpy as np
import os

# dataset = None


def create_augmented_alignment():
  dataset.init_seq_order()
  seq_idx = 0
  seg_total_len = 0
  num_segs = 0
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx < 3:
    if seq_idx % 1000 == 0:
      complete_frac = dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    dataset.load_seqs(seq_idx, seq_idx + 1)
    orth = dataset.get_data(seq_idx, "data")
    bpe_align = dataset.get_data(seq_idx, "bpe_align")
    phoneme_align = dataset.get_data(seq_idx, "phoneme_align")
    print(orth)
    print(bpe_align)
    print(phoneme_align)

    seq_idx += 1


def init(bpe_hdf, phoneme_hdf, ):
  global config
  global dataset
  global bpe_vocab
  global phoneme_vocab
  global phoneme_lexicon

  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  rnn.init_config(config_filename=None, default_config={"cache_size": 0})
  config = rnn.config
  config.set("log", None)
  rnn.init_log()
  print("Returnn augment bpe align starting up", file=rnn.log.v2)

  # init bpe and phoneme align datasets
  bpe_dataset_dict = {
    "class": "HDFDataset", "files": [bpe_hdf], "use_cache_manager": True, 'estimated_num_seqs': 3000,
    'partition_epoch': 1,
    'seq_list_filter_file': '/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000'}
  phoneme_dataset_dict = {
    "class": "HDFDataset", "files": [phoneme_hdf], "use_cache_manager": True, 'estimated_num_seqs': 3000,
    'partition_epoch': 1,
    'seq_list_filter_file': '/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000'}
  sprint_dataset_dict = {
      'class': 'ExternSprintDataset', 'input_stddev': 3.0, 'orth_vocab': None,
      'sprintConfigStr': ['--config=/u/schmitt/experiments/transducer/config/rasr-configs/dev_data.sprint.config'],
      'sprintTrainerExecPath': '/u/schmitt/experiments/transducer/config/sprint-executables/nn-trainer',
      'suppress_load_seqs_print': True}

  dataset_dict = {
    'class': 'MetaDataset',
    'data_map':
      {'bpe_align': ('bpe_align', 'data'), 'phoneme_align': ('phoneme_align', 'data'), 'data': ('sprint', 'orth_classes')},
    'datasets': {
      'bpe_align': bpe_dataset_dict, "phoneme_align": phoneme_dataset_dict, 'sprint': sprint_dataset_dict},
    'seq_order_control_dataset': 'bpe_align'}

  dataset = rnn.init_dataset(dataset_dict)

  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()


def main():
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("bpe_align_hdf", help="hdf file which contains the extracted bpe alignments")
  arg_parser.add_argument("phoneme_align_hdf", help="hdf file which contains the extracted phoneme alignments")
  arg_parser.add_argument("--bpe_blank_idx", help="the blank index in the bpe alignment", type=int)
  arg_parser.add_argument("--phoneme_blank_idx", help="the blank index in the phoneme alignment", type=int)
  arg_parser.add_argument("--bpe_vocab", help="mapping from bpe idx to label", type=str)
  arg_parser.add_argument("--phoneme_vocab", help="mapping from phoneme idx to label", type=str)
  arg_parser.add_argument("--phoneme_lexicon", help="mapping from words to phonemes", type=str)
  arg_parser.add_argument("--corpus", help="path to the corpus file", type=str)
  arg_parser.add_argument("--out_path", help="output path for augmented alignment", type=str)
  args = arg_parser.parse_args()
  global rnn
  import returnn.__main__ as rnn

  init(args.bpe_align_hdf, args.phoneme_align_hdf)

  try:
    create_augmented_alignment()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
