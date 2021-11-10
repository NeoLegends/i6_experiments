import argparse
import sys
import numpy as np
import os

# dataset = None


def calc_segment_stats(blank_idx, segment):
  if segment == "total":
    segment_idxs = ":"
    initial_idx = "0"
  elif segment == "first":
    segment_idxs = ":1"
    initial_idx = "0"
  elif segment == "except_first":
    segment_idxs = "1:"
    initial_idx = "non_blank_idxs[0]"
  else:
    raise ValueError("segment definition unknown")
  dataset.init_seq_order()
  seq_idx = 0
  seg_total_len = 0
  num_segs = 0
  while dataset.is_less_than_num_seqs(seq_idx):
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "data")
    # print(data)
    non_blank_idxs = np.where(data != blank_idx)[0]
    if non_blank_idxs.size == 0:
      continue
    else:
      # non_blank_idxs = np.append(non_blank_idxs)
      prev_i = eval(initial_idx)
      try:
        for i in eval("non_blank_idxs[" + segment_idxs + "]"):
          # each non-blank idx corresponds to one segment
          num_segs += 1
          # the segment length is the difference from the previous border to the current one
          # for the first and last segment, the result needs to be corrected (see after while)
          seg_total_len += i - prev_i
          prev_i = i
      except IndexError:
        continue

    seq_idx += 1

  # the first segment is always 1 too short
  if segment == "first":
    seg_total_len += num_segs

  mean_seg_len = seg_total_len / num_segs

  filename = "mean.seg.len"
  mode = "w"
  if os.path.exists(filename):
    mode = "a"
  with open(filename, mode) as f:
    f.write(segment + ": " + str(mean_seg_len) + "\n")


def init(hdf_file, seq_list_filter_file):
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  dataset_dict = {
    "class": "HDFDataset", "files": [hdf_file], "use_cache_manager": True, "seq_list_filter_file": seq_list_filter_file
  }

  rnn.init_config(config_filename=None, default_config={"cache_size": 0})
  global config
  config = rnn.config
  config.set("log", None)
  global dataset
  dataset = rnn.init_dataset(dataset_dict)
  rnn.init_log()
  print("Returnn segment-statistics starting up", file=rnn.log.v2)
  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()


def main():
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("hdf_file", help="hdf file which contains the extracted alignments of some corpus")
  arg_parser.add_argument("--seq-list-filter-file", help="whitelist of sequences to use", default=None)
  arg_parser.add_argument("--blank-idx", help="the blank index in the alignment", default=0, type=int)
  arg_parser.add_argument("--segment", help="over which segments to calculate the statistics: 'total', 'first', "
                                            "'except_first', 'all' (default: 'all')", default="all")
  arg_parser.add_argument("--returnn-root", help="path to returnn root")
  args = arg_parser.parse_args()
  assert args.segment in ["first", "except_first", "all", "total"]
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn

  init(args.hdf_file, args.seq_list_filter_file)

  try:
    if args.segment == "all":
      for seg in ["total", "first", "except_first"]:
        calc_segment_stats(args.blank_idx, seg)
    else:
      calc_segment_stats(args.blank_idx, args.segment)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
