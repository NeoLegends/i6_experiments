import argparse
import sys
import numpy as np

def calc_segment_stats_with_sil(blank_idx, sil_idx):

  dataset.init_seq_order()
  seq_idx = 0
  inter_sil_seg_len = 0
  init_sil_seg_len = 0
  final_sil_seg_len = 0
  label_seg_len = 0
  num_blank_frames = 0
  num_label_segs = 0
  num_sil_segs = 0
  num_init_sil_segs = 0
  num_final_sil_segs = 0
  num_seqs = 0

  while dataset.is_less_than_num_seqs(seq_idx):
    num_seqs += 1
    # progress indication
    if seq_idx % 1000 == 0:
      complete_frac = dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "data")

    non_blank_idxs = np.where(data != blank_idx)[0]
    sil_idxs = np.where(data == sil_idx)[0]
    num_label_segs += len(non_blank_idxs) - len(sil_idxs)
    num_sil_segs += len(sil_idxs)
    num_blank_frames += len(data) - len(non_blank_idxs)

    if non_blank_idxs.size == 0:
      seq_idx += 1
      continue
    else:
      prev_idx = 0
      try:
        for i, idx in enumerate(non_blank_idxs):
          seg_len = idx - prev_idx
          if prev_idx == 0:
            seg_len += 1

          if idx in sil_idxs:
            if i == 0:
              init_sil_seg_len += seg_len
              num_init_sil_segs += 1
            elif i == len(non_blank_idxs) - 1:
              final_sil_seg_len += seg_len
              num_final_sil_segs += 1
            else:
              inter_sil_seg_len += seg_len
          else:
            label_seg_len += seg_len

          prev_idx = idx
      except IndexError:
        continue

    seq_idx += 1

  mean_init_sil_len = init_sil_seg_len / num_init_sil_segs if num_init_sil_segs > 0 else 0
  mean_final_sil_len = final_sil_seg_len / num_final_sil_segs if num_final_sil_segs > 0 else 0
  mean_inter_sil_len = inter_sil_seg_len / (num_sil_segs - num_init_sil_segs - num_final_sil_segs) if inter_sil_seg_len > 0 else 0
  mean_total_sil_len = (init_sil_seg_len + final_sil_seg_len + inter_sil_seg_len) / num_seqs

  mean_label_len = label_seg_len / num_label_segs

  mean_seq_len = (num_blank_frames + num_sil_segs + num_label_segs) / num_seqs

  filename = "statistics"
  with open(filename, "w+") as f:
    f.write("Segment statistics: \n\n")
    f.write("\tSilence: \n")
    f.write("\t\tInitial:\n")
    f.write("\t\t\tMean length: %f \n" % mean_init_sil_len)
    f.write("\t\t\tNum segments: %f \n" % num_init_sil_segs)
    f.write("\t\tIntermediate:\n")
    f.write("\t\t\tMean length: %f \n" % mean_inter_sil_len)
    f.write("\t\t\tNum segments: %f \n" % (num_sil_segs - num_init_sil_segs - num_final_sil_segs))
    f.write("\t\tFinal:\n")
    f.write("\t\t\tMean length: %f \n" % mean_final_sil_len)
    f.write("\t\t\tNum segments: %f \n" % num_final_sil_segs)
    f.write("\t\tTotal:\n")
    f.write("\t\t\tMean length: %f \n" % mean_total_sil_len)
    f.write("\t\t\tNum segments: %f \n" % num_sil_segs)
    f.write("\n")
    f.write("\tNon-silence: \n")
    f.write("\t\tMean length: %f \n" % mean_label_len)
    f.write("\t\tNum segments: %f \n" % num_label_segs)
    f.write("\n")
    f.write("\n")
    f.write("Sequence statistics: \n\n")
    f.write("\tMean length: %f \n" % mean_seq_len)
    f.write("\tNum sequences: %f \n" % num_seqs)

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
  arg_parser.add_argument("--sil-idx", help="the blank index in the alignment", default=None, type=int)
  arg_parser.add_argument("--returnn-root", help="path to returnn root")
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn

  init(args.hdf_file, args.seq_list_filter_file)

  try:
    calc_segment_stats_with_sil(args.blank_idx, args.sil_idx)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
