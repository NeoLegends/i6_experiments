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
import tensorflow as tf

# dataset = None


def hdf_dataset_init(out_dim, file_name):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename=file_name, dim=out_dim, ndim=1)


def create_augmented_alignment(bpe_upsampling_factor, hdf_dataset):
  dataset.init_seq_order()
  seq_idx = 0
  sil_idx = 0
  special_tokens = ("[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]")
  skipped_pairs = []
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx < float("inf"):
    skip_seq = False
    if seq_idx % 1000 == 0:
      complete_frac = dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    dataset.load_seqs(seq_idx, seq_idx + 1)

    # load alignments (idx sequences)
    bpe_align = dataset.get_data(seq_idx, "bpe_align")
    phoneme_align = dataset.get_data(seq_idx, "phoneme_align")
    # bpe and phoneme string sequence
    bpes = np.array([bpe_vocab[idx] for idx in bpe_align])
    phonemes = np.array([phoneme_vocab[idx] for idx in phoneme_align])
    # string seqs without blanks
    bpes_non_blank = bpes[bpe_align != bpe_blank_idx]
    phonemes_non_blank = phonemes[phoneme_align != phoneme_blank_idx]
    # upscale bpe sequence to match phoneme align length
    rem_num = len(phoneme_align) % bpe_upsampling_factor
    upscaled_bpe_align = [i for j in bpe_align[:-1] for i in ([bpe_blank_idx] * (bpe_upsampling_factor-1)) + [j]]
    if rem_num == 0:
      upscaled_bpe_align += [i for j in bpe_align[-1:] for i in ([bpe_blank_idx] * (bpe_upsampling_factor-1)) + [j]]
    else:
      upscaled_bpe_align += [i for j in bpe_align[-1:] for i in ([bpe_blank_idx] * (rem_num - 1)) + [j]]
    # get word sequence by merging non-blank bpe strings
    words = []
    cur = ""
    for subword in bpes_non_blank:
      cur += subword
      if subword.endswith("@@"):
        cur = cur[:-2]
      else:
        words.append(cur)
        cur = ""

    # get a unique mapping from words to phonemes
    # if a unique mapping is not possible, store and skip the sequence
    word_phon_map = []
    rem_phons = " ".join(phonemes_non_blank[phonemes_non_blank != "[SILENCE]"])
    for word in words:
      if word in special_tokens:
        word_phon_map.append(word)
        rem_phons = rem_phons[len(word + " "):]
      else:
        # find the lemma node in the lexicon which has the current word as orthography
        lemma = [element for element in lexicon.iter("lemma") if element.find("orth").text == word][0]
        phon_cands = [phon.text for phon in lemma.findall("phon")]
        matching_cands = []
        for cand in phon_cands:
          if rem_phons.startswith(cand):
            matching_cands.append(cand)
        if len(matching_cands) != 1:
          skip_seq = True
          break
        word_phon_map.append(matching_cands[0])
        rem_phons = rem_phons[len(matching_cands[0] + " "):]
    if skip_seq:
      seq_idx += 1
      skipped_pairs.append((bpes_non_blank, phonemes_non_blank))
      continue

    # get mapping from word sequence to bpe tokens
    # additionally, store the fraction of a subword of the merged word
    # e.g.: for [cat@@, s], store the fraction that "cat@@" and "s" inhabit in the total alignment of "cats"
    word_bpe_map = []
    mapping = []
    prev_bound = 0
    for i, bpe_idx in enumerate(upscaled_bpe_align):
      if bpe_idx != bpe_blank_idx:
        seg_size = i - prev_bound
        prev_bound += seg_size
        if prev_bound == seg_size:
          seg_size += 1
        if bpe_vocab[bpe_idx].endswith("@@"):
          mapping.append([bpe_idx, seg_size])
        else:
          mapping.append([bpe_idx, seg_size])
          total_size = sum(size for _, size in mapping)
          mapping = [[label, size / total_size] for label, size in mapping]
          word_bpe_map.append(mapping)
          mapping = []

    # determine the word boundaries in the phoneme alignment
    word_bounds = []
    sil_bounds = []
    new_bpe_align = []
    align_idx = 0
    # go through each word to phoneme mapping
    for mapping in word_phon_map:
      # print(mapping)
      word_phons = mapping.split(" ")
      # word_phons = [phon_to_idx[phon] for phon in word_phons]
      last_phon_in_word = word_phons[-1]
      if last_phon_in_word not in special_tokens:
        last_phon_in_word += "{#+#}@f.0"
      else:
        last_phon_in_word += "{#+#}.0"
      last_phon_in_word = phon_to_idx[last_phon_in_word]
      # go through the phoneme align, starting from the word boundary of the previous word
      for i, phon_idx in enumerate(phoneme_align[align_idx:]):
        if phon_idx == sil_idx:
          word_bounds.append(align_idx + i)
          sil_bounds.append(align_idx + i)
        # store word end positions, update the align_idx and go to the next word mapping
        elif phon_idx == last_phon_in_word:
          word_bounds.append(align_idx + i)
          align_idx = align_idx + i + 1
          break
    if len(phoneme_align[align_idx:]) > 0:
      assert phoneme_align[-1] == sil_idx
      sil_bounds.append(len(phoneme_align)-1)
      word_bounds.append(len(phoneme_align) - 1)

    new_bpe_blank_idx = bpe_blank_idx + 1
    bpe_sil_align = [new_bpe_blank_idx] * len(phoneme_align)
    prev_bound = -1
    bpe_idx = 0
    print(seq_idx)
    for bound in word_bounds:
      if bound in sil_bounds:
        bpe_sil_align[bound] = sil_idx
        prev_bound = bound
      else:
        size = bound - prev_bound
        if prev_bound == 0:
          size += 1
        bpe_map = word_bpe_map[bpe_idx]
        for i, (bpe, frac) in enumerate(bpe_map):
          if i != len(bpe_map) - 1:
            offset = max(int(round(size * frac, 0)), 1)
            bpe_sil_align[prev_bound + offset] = bpe
            prev_bound += offset
          else:
            bpe_sil_align[bound] = bpe
        bpe_idx += 1
        prev_bound = bound

    # plot some random examples
    if np.random.rand(1) < 0.01:
      plot_aligns(upscaled_bpe_align, phoneme_align, bpe_sil_align, seq_idx)

    # dump new alignment into hdf file
    seq_len = len(bpe_sil_align)
    tag = dataset.get_tag(seq_idx)
    new_data = tf.constant(np.expand_dims(bpe_sil_align, axis=0), dtype="int32")
    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1  # - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    hdf_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)

    seq_idx += 1

  print("Skipped Sequence Pairs:")
  print("\n".join([str(pair) for pair in skipped_pairs]))


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
  bpe_align = np.array(bpe_align)
  bpe_silence_align = np.array(bpe_silence_align)
  phoneme_align = np.array(phoneme_align)

  bpe_ticks = np.where(bpe_align != bpe_blank_idx)[0]
  bpe_labels = bpe_align[bpe_align != bpe_blank_idx]
  bpe_labels = [bpe_vocab[i] for i in bpe_labels]

  bpe_silence_ticks = np.where(bpe_silence_align != bpe_blank_idx + 1)[0]
  bpe_silence_labels = bpe_silence_align[bpe_silence_align != bpe_blank_idx + 1]
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


def init(
  bpe_hdf, phoneme_hdf, bpe_vocab_file, phoneme_vocab_file, phoneme_lexicon_file, bpe_blank, phoneme_blank,
  segment_file, out_vocab):
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
    bpe_silence_vocab = bpe_vocab.copy()
    voc_noise_idx = bpe_vocab.pop("[VOCALIZED-NOISE]")
    bpe_vocab["[VOCALIZEDNOISE]"] = voc_noise_idx
    bpe_vocab = {int(v): k for k, v in bpe_vocab.items()}
    bpe_vocab[bpe_blank] = "<b>"

  bpe_silence_vocab["<s>"] = bpe_blank
  bpe_silence_vocab["</s>"] = bpe_blank
  bpe_silence_vocab["[SILENCE]"] = 0
  with open(out_vocab, "w+") as f:
    json.dump(bpe_silence_vocab, f)
  bpe_silence_vocab["<b>"] = bpe_blank + 1
  voc_noise_idx = bpe_silence_vocab.pop("[VOCALIZED-NOISE]")
  bpe_silence_vocab["[VOCALIZEDNOISE]"] = voc_noise_idx
  bpe_silence_vocab = {int(v): k for k, v in bpe_silence_vocab.items()}

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
    'seq_list_filter_file': segment_file}
  phoneme_dataset_dict = {
    "class": "HDFDataset", "files": [phoneme_hdf], "use_cache_manager": True, 'estimated_num_seqs': 3000,
    'partition_epoch': 1,
    'seq_list_filter_file': segment_file}

  dataset_dict = {
    'class': 'MetaDataset',
    'data_map':
      {'bpe_align': ('bpe_align', 'data'), 'phoneme_align': ('phoneme_align', 'data')},
    'datasets': {
      'bpe_align': bpe_dataset_dict, "phoneme_align": phoneme_dataset_dict},
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
  arg_parser.add_argument("--segment_file", help="segment whitelist", type=str)
  arg_parser.add_argument(
    "--bpe_upsampling_factor", help="factor to get bpe alignment to same length as phoneme alignment", type=int)
  arg_parser.add_argument("--out_align", help="output path for augmented alignment", type=str)
  arg_parser.add_argument("--out_vocab", help="output path for augmented vocab", type=str)
  arg_parser.add_argument("--returnn_root", type=str)
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  init(
    args.bpe_align_hdf, args.phoneme_align_hdf, args.bpe_vocab, args.phoneme_vocab, args.phoneme_lexicon,
    args.bpe_blank_idx, args.phoneme_blank_idx, args.segment_file, args.out_vocab)

  hdf_dataset = hdf_dataset_init(out_dim=dataset.get_data_dim("bpe_align") + 1, file_name=args.out_align)

  try:
    create_augmented_alignment(args.bpe_upsampling_factor, hdf_dataset)
    hdf_dataset.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
