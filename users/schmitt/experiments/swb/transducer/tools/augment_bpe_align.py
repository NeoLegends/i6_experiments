import argparse
import ast
import json
import sys
import numpy as np
import os
import xml, gzip
import xml.etree.ElementTree as ET
from xml import etree
import matplotlib.pyplot as plt
import numpy as np

# dataset = None


def create_augmented_alignment():
  dataset.init_seq_order()
  seq_idx = 0
  seg_total_len = 0
  sil_idx = 0
  num_segs = 0
  num_plotted = 0
  # phon_to_idx = {v: k for k, v in phoneme_vocab.items()}
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx < float("inf"):
    if seq_idx % 1000 == 0:
      complete_frac = dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    dataset.load_seqs(seq_idx, seq_idx + 1)
    orth = dataset.get_data(seq_idx, "data")
    bpe_align = dataset.get_data(seq_idx, "bpe_align")
    phoneme_align = dataset.get_data(seq_idx, "phoneme_align")
    # if len(orth) > 5:
    words = [word_vocab[idx] for idx in orth]
    # print(words)
    bpes = [bpe_vocab[idx] for idx in bpe_align]
    # print(bpes)
    phonemes = [phoneme_vocab[idx] for idx in phoneme_align]
    # print(phoneme_align)
    print([word_vocab[idx] for idx in orth])
    word_phon_map = [[element for element in lexicon.iter("lemma") if element.find("orth").text == word_vocab[idx]][0].find("phon").text for idx in orth if word_vocab[idx] != "<unk>"]
    print(word_phon_map)
    # print(word_phon_map)

    rem_num = len(phoneme_align) % 6
    upscaled_bpe_align = [i for j in bpe_align[:-1] for i in ([bpe_blank_idx] * 5) + [j]]
    if rem_num == 0:
      upscaled_bpe_align += [i for j in bpe_align[-1:] for i in ([bpe_blank_idx] * 5) + [j]]
    else:
      upscaled_bpe_align += [i for j in bpe_align[-1:] for i in ([bpe_blank_idx] * (rem_num - 1)) + [j]]

    word_bpe_map = []
    mapping = []
    prev_bound = 0
    for i, bpe_idx in enumerate(upscaled_bpe_align):
      if bpe_idx != bpe_blank_idx:
        seg_size = i - prev_bound
        prev_bound += seg_size
        if prev_bound == seg_size:
          seg_size += 1
          # prev_bound += seg_size - 1
        if bpe_vocab[bpe_idx].endswith("@@"):
          mapping.append([bpe_idx, seg_size])
        else:
          mapping.append([bpe_idx, seg_size])
          total_size = sum(size for _, size in mapping)
          mapping = [[label, size / total_size] for label, size in mapping]
          word_bpe_map.append(mapping)
          mapping = []

    # print("UPSCALED BPE ALIGN: ", upscaled_bpe_align)
    # print("WORD BPE MAP: ", word_bpe_map)

    word_bounds = []
    sil_bounds = []
    new_bpe_align = []
    align_idx = 0
    # go through each word to phoneme mapping
    for mapping in word_phon_map:
      # print(mapping)
      word_phons = mapping.split(" ")
      # word_phons = [phon_to_idx[phon] for phon in word_phons]
      last_phon_in_word = word_phons[-1] + "{#+#}@f.0"
      last_phon_in_word = phon_to_idx[last_phon_in_word]
      # go through the phoneme align, starting from the word boundary of the previous word
      # print("START ALIGN IDX: ", align_idx)
      # print("LAST PHON IDX: ", last_phon_in_word)
      # print("REM ALIGN: ", phoneme_align[align_idx:])
      for i, phon_idx in enumerate(phoneme_align[align_idx:]):
        if phon_idx == sil_idx:
          word_bounds.append(align_idx + i)
          sil_bounds.append(align_idx + i)
        # store word end positions, update the align_idx and go to the next word mapping
        elif phon_idx == last_phon_in_word:
          word_bounds.append(align_idx + i)
          align_idx = align_idx + i + 1
          break

    # print("WORD BOUNDS: ", word_bounds)
    # print("SIL BOUNDS: ", sil_bounds)

    bpe_sil_align = [bpe_blank_idx] * len(phoneme_align)
    prev_bound = -1
    bpe_idx = 0
    for bound in word_bounds:
      if bound in sil_bounds:
        bpe_sil_align[bound] = sil_idx
        prev_bound = bound
      else:
        size = bound - prev_bound
        if prev_bound == 0:
          size += 1
        bpe_map = word_bpe_map[bpe_idx]
        # print("BPE MAP: ", bpe_map)
        # print("BOUND: ", bound)
        for bpe, frac in bpe_map:
          # print("PREV BOUND: ", prev_bound)
          bpe_sil_align[prev_bound + int(round(size * frac, 0))] = bpe
          prev_bound += int(round(size * frac, 0))
        bpe_idx += 1
        prev_bound = bound

    # print("BPE SIL ALIGN: ", bpe_sil_align)

    # plot_aligns(upscaled_bpe_align, phoneme_align, bpe_sil_align, seq_idx)
    # num_plotted += 1

    # if num_plotted > 0:
    #   break

    seq_idx += 1


def plot_aligns(bpe_align, phoneme_align, bpe_silence_align, seq_idx):
  # rem_num = len(phoneme_align) % 6
  # bpe_align = [
  #   i for j in red_bpe_align[:-1] for i in ([bpe_blank_idx] * 5) + [j]]
  # print(bpe_align)
  # if rem_num == 0:
  #   bpe_align += [i for j in red_bpe_align[-1:] for i in ([bpe_blank_idx] * 5) + [j]]
  # else:
  #   bpe_align += [i for j in red_bpe_align[-1:] for i in ([bpe_blank_idx] * (rem_num - 1)) + [j]]
  matrix = np.concatenate(
    [np.array([[0 if i == bpe_blank_idx else 1 for i in bpe_align]]),
     np.array([[0 if i == bpe_blank_idx else 1 for i in bpe_silence_align]]),
     np.array([[0 if i == phoneme_blank_idx else 1 for i in phoneme_align]])],
    axis=0
  )
  print(bpe_align)
  bpe_align = np.array(bpe_align)
  bpe_silence_align = np.array(bpe_silence_align)
  phoneme_align = np.array(phoneme_align)

  bpe_ticks = np.where(bpe_align != bpe_blank_idx)[0]
  bpe_labels = bpe_align[bpe_align != bpe_blank_idx]
  bpe_labels = [bpe_vocab[i] for i in bpe_labels]

  bpe_silence_ticks = np.where(bpe_silence_align != bpe_blank_idx)[0]
  bpe_silence_labels = bpe_silence_align[bpe_silence_align != bpe_blank_idx]
  bpe_silence_labels = [bpe_silence_vocab[i] for i in bpe_silence_labels]

  phoneme_ticks = np.where(phoneme_align != phoneme_blank_idx)[0]
  phoneme_labels = phoneme_align[phoneme_align != phoneme_blank_idx]
  phoneme_labels = [phoneme_vocab[i] for i in phoneme_labels]

  plt.figure(figsize=(10, 5), constrained_layout=True)
  ax = plt.gca()
  # fig = plt.gcf()
  # fig.set_size_inches(10, 2)
  matshow = ax.matshow(matrix, aspect="auto", cmap=plt.cm.get_cmap("Blues"))
  # # create second x axis for hmm alignment labels and plot same matrix
  hmm_ax = ax.twiny()
  bpe_silence_ax = ax.twiny()

  #
  ax.set_xticks(list(bpe_ticks))
  # ax.xaxis.set_major_formatter(ticker.NullFormatter())
  # ax.set_xticks(bpe_xticks_minor, minor=True)
  ax.set_xticklabels(list(bpe_labels))
  # ax.tick_params(axis="x", which="minor", length=0, labelsize=17)
  # ax.tick_params(axis="x", which="major", length=10)
  ax.set_xlabel("BPE Alignment")
  # ax.set_ylabel("Output RNA BPE Labels", fontsize=18)
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  # ax.spines['top'].set_position(('outward', 50))
  # # for tick in ax.xaxis.get_minor_ticks():
  # #   tick.label1.set_horizontalalignment("center")

  bpe_silence_ax.set_xlim(ax.get_xlim())
  bpe_silence_ax.set_xticks(list(bpe_silence_ticks))
  bpe_silence_ax.set_xticklabels(list(bpe_silence_labels))
  # bpe_silence_ax.xaxis.tick_top()
  # bpe_silence_ax.set_xlabel("BPE + Silence Alignment")
  bpe_silence_ax.xaxis.set_label_position('top')
  bpe_silence_ax.spines['top'].set_position(('outward', 50))

  # # set x ticks and labels and positions for hmm axis
  hmm_ax.set_xticks(phoneme_ticks)
  # hmm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
  # hmm_ax.set_xticks(hmm_xticks_minor, minor=True)
  hmm_ax.set_xticklabels(phoneme_labels, rotation="vertical")
  # hmm_ax.tick_params(axis="x", which="minor", length=0)
  # hmm_ax.tick_params(axis="x", which="major", length=0)
  hmm_ax.xaxis.set_ticks_position('bottom')
  hmm_ax.xaxis.set_label_position('bottom')
  hmm_ax.set_xlabel("HMM Phoneme Alignment")
  #
  # time_xticks = [x - .5 for x in range(0, len(phoneme_align), 2)]
  # time_xticks_labels = [x for x in range(0, len(phoneme_align), 2)]
  # time_ax.set_xlabel("Input Time Frames", fontsize=18)
  # time_ax.xaxis.tick_bottom()
  # time_ax.xaxis.set_label_position('bottom')
  # time_ax.set_xlim(ax.get_xlim())
  # time_ax.set_xticks(time_xticks)
  # time_ax.set_xticklabels(time_xticks_labels, fontsize=17)

  plt.savefig("plot%s.png" % seq_idx)
  plt.close()


def init(bpe_hdf, phoneme_hdf, bpe_vocab_file, phoneme_vocab_file, phoneme_lexicon_file, bpe_blank, phoneme_blank):
  global config
  global dataset
  global word_vocab
  global bpe_vocab
  global bpe_silence_vocab
  global phoneme_vocab
  global lexicon
  global bpe_blank_idx
  global phoneme_blank_idx
  global phon_to_idx

  bpe_blank_idx = bpe_blank
  phoneme_blank_idx = phoneme_blank

  with open(bpe_vocab_file, "r") as f:
    bpe_vocab = ast.literal_eval(f.read())
    bpe_vocab = {int(v): k for k, v in bpe_vocab.items()}
    bpe_vocab[bpe_blank] = "<b>"
  bpe_silence_vocab = bpe_vocab.copy()
  bpe_silence_vocab[bpe_blank + 1] = "[SILENCE]"
  with open(phoneme_vocab_file, "r") as f:
    phon_to_idx = ast.literal_eval(f.read())
    phon_to_idx = {k: int(v) for k, v in phon_to_idx.items()}
    phoneme_vocab = {int(v): k.split("{")[0] for k, v in phon_to_idx.items()}
    phoneme_vocab[phoneme_blank] = "<b>"
    phon_to_idx["<b>"] = phoneme_blank
  word_vocab_file = "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/words/vocab_json"
  with open(word_vocab_file, "r") as f:
    word_vocab = ast.literal_eval(f.read())
    word_vocab = {int(v): k for k, v in word_vocab.items()}
  with gzip.open(phoneme_lexicon_file, "r") as f:
    # xml_parser = ET.XMLParser(encoding="iso-8859-5")
    lexicon = ET.fromstring(f.read())


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
      'class': 'ExternSprintDataset', 'input_stddev': 3.0,
      'sprintConfigStr': ['--config=/u/schmitt/experiments/transducer/config/rasr-configs/dev_data.sprint.config'],
      'sprintTrainerExecPath': '/u/zhou/rasr-dev/arch/linux-x86_64-standard-label_sync_decoding/nn-trainer.linux-x86_64-standard-label_sync_decoding',
      "orth_vocab": {
        "vocab_file": "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/words/vocab_json",
        "unknown_label": "<unk>"},
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
  arg_parser.add_argument("--returnn_root", type=str)
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn

  init(
    args.bpe_align_hdf, args.phoneme_align_hdf, args.bpe_vocab, args.phoneme_vocab, args.phoneme_lexicon,
    args.bpe_blank_idx, args.phoneme_blank_idx)

  try:
    create_augmented_alignment()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
