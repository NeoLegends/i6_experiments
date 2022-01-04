#!/usr/bin/env python

"""
For debugging, go through some dataset, forward it through the net, and output the layer activations on stdout.
"""

from __future__ import print_function

import sys
import numpy as np
import os
import ast
import subprocess
import tensorflow as tf

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
import argparse
from matplotlib import pyplot as plt
from matplotlib import ticker
from returnn.util.basic import pretty_print


def dump(dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  print("Epoch: %i" % options.epoch, file=log.v3)
  dataset.init_seq_order(options.epoch)
  print("LABELS: ", dataset.labels)

  layers = ["encoder"]
  rec_layers = ["att_weights", "att_query", "att_ctx", "att_val", "att", "att_energy", "output"]
  if rnn.config.typed_value("att_area") == "seg":
    rec_layers.append("segment_starts")
    # rec_layers.append("att_unmasked")
  np.set_printoptions(threshold=np.inf)

  if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)

  output_dict = {}
  # get the output of layers outside the rec loop
  for name in layers:
    layer = rnn.engine.network.get_layer(name)
    output_dict["%s-out" % name] = layer.output.get_placeholder_as_batch_major()
    with open(os.path.join(options.out_dir, name + "-shape"), "w+") as f:
      f.write(repr(layer.output.dim_tags))
  # get the output of layers inside the rec loop
  for name in rec_layers:
    layer = rnn.engine.network.get_layer("output/" + name)
    output_dict["%s-out" % name] = layer.output.get_placeholder_as_batch_major()
    with open(os.path.join(options.out_dir, name + "-shape"), "w+") as f:
      f.write(repr(layer.output.dim_tags))
  # get additional information
  output_dict["output_len"] = rnn.engine.network.get_layer("output").output.get_sequence_lengths()
  output_dict["encoder_len"] = rnn.engine.network.get_layer("encoder").output.get_sequence_lengths()
  output_dict["seq_idx"] = rnn.engine.network.get_extern_data("seq_idx").get_placeholder_as_batch_major()
  output_dict["seq_tag"] = rnn.engine.network.get_extern_data("seq_tag").get_placeholder_as_batch_major()
  output_dict["target_data"] = rnn.engine.network.get_extern_data(
    rnn.engine.network.extern_data.default_input).get_placeholder_as_batch_major()
  output_dict["target_classes"] = rnn.engine.network.get_extern_data("targetb").get_placeholder_as_batch_major()

  # get attention weight axis information
  weight_layer = rnn.engine.network.get_layer("output/att_weights").output.copy_as_batch_major()
  weight_batch_axis = weight_layer.batch_dim_axis
  weight_time_axis = weight_layer.time_dim_axis
  weight_head_axis = weight_layer.get_axis_from_description("stag:att_heads")
  weight_att_axis = weight_layer.get_axis_from_description("stag:att_t")

  energy_layer = rnn.engine.network.get_layer("output/att_energy").output.copy_as_batch_major()
  energy_batch_axis = energy_layer.batch_dim_axis
  energy_time_axis = energy_layer.time_dim_axis
  energy_head_axis = energy_layer.get_axis_from_description("stag:att_heads")
  energy_att_axis = energy_layer.get_axis_from_description("stag:att_t")

  if rnn.config.typed_value("att_area") == "seg":
    output_dict["energy_len"] = rnn.engine.network.get_layer("output/att_energy").output.dim_tags[energy_att_axis].dyn_size

  # go through the specified sequences and dump the outputs from output_dict into separate files
  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)
  plot_dir = os.path.join(options.out_dir, "att_plots")
  if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= options.endseq:
    out = rnn.engine.run_single(dataset=dataset, seq_idx=seq_idx, output_dict=output_dict)
    # subfolder = os.path.join(options.out_dir, "sequence-%d" % seq_idx)
    # if not os.path.exists(subfolder):
    #   os.mkdir(subfolder)
    # dump all outputs
    # for name, v in sorted(out.items()):
      # with open(os.path.join(subfolder, name), "w+") as f:
      #   f.write("SHAPE: " + repr(v.shape) + "\n")
      #   f.write(repr(v))

    seq_tag = out["seq_tag"]
    encoder_len = out["encoder_len"][0]

    hmm_align = subprocess.check_output(
      ["/u/schmitt/experiments/transducer/config/sprint-executables/archiver", "--mode", "show", "--type", "align",
        "--allophone-file", "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/allophones",
        "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/tuske__2016_01_28__align.combined.train",
        seq_tag[0]])
    hmm_align = hmm_align.splitlines()
    hmm_align = [row.decode("utf-8").strip() for row in hmm_align]
    hmm_align = [row for row in hmm_align if row.startswith("time")]
    hmm_align = [row.split("\t")[5].split("{")[0] for row in hmm_align]
    hmm_align = [phon if phon != "[SILENCE]" else "[S]" for phon in hmm_align]

    # hmm_align = list(np.random.randint(0, 10, (6 * encoder_len - 3,)))

    hmm_align_borders = [i+1 for i, x in enumerate(hmm_align) if i < len(hmm_align)-1 and hmm_align[i] != hmm_align[i+1]]
    if hmm_align_borders[-1] != len(hmm_align):
      hmm_align_borders += [len(hmm_align)]
    hmm_align_major = [hmm_align[i-1] for i in range(1, len(hmm_align) + 1) if i in hmm_align_borders]
    if hmm_align_borders[0] != 0:
      hmm_align_borders = [0] + hmm_align_borders
    # print(hmm_align)
    # print(len(hmm_align))
    # print(hmm_align_major)
    # print(hmm_align_borders)
    # hmm_align_major = [hmm_align[i] for i in range(len(hmm_align)) if i in hmm_align_borders]
    hmm_align_borders_center = [(i+j) / 2 for i, j in zip(hmm_align_borders[:-1], hmm_align_borders[1:])]

    last_label_rep = len(hmm_align) - (encoder_len-1) * 6
    target_align = out["target_classes"][0][:]

    targetb_blank_idx = rnn.config.typed_value("targetb_blank_idx")

    weights = np.transpose(
      out["att_weights-out"],
      (weight_batch_axis, weight_time_axis, weight_att_axis, weight_head_axis))

    energies = np.transpose(
      out["att_energy-out"],
      (energy_batch_axis, energy_time_axis, energy_att_axis, energy_head_axis))

    fig, axs = plt.subplots(2, weights.shape[3], figsize=(weights.shape[3] * 30, 35), constrained_layout=True)

    for head in range(weights.shape[3]):
      weights_ = weights[0, :, :, head]  # (B, dec, enc, heads) -> (dec, enc)
      energies_ = energies[0, :, :, head]
      if rnn.config.typed_value("att_area") == "win" and rnn.config.typed_value("att_win_size") == "full":
        pass

      elif rnn.config.typed_value("att_area") == "win":
        excess_size = int(weights_.shape[1] / 2)
        zeros = np.zeros(
          (weights_.shape[0], (encoder_len - weights_.shape[1]) + excess_size * 2))  # (dec, (enc - win) + 2*(win//2))
        ones = np.ones_like(zeros)
        # append zeros to the weights to fill up the whole encoder length
        weights_ = np.concatenate([weights_, zeros], axis=1)  # (dec, enc + 2*(win//2))
        energies_ = np.concatenate([energies_, ones * np.min(energies_)], axis=1)
        # roll the weights to their corresponding position in the encoder sequence (sliding window)
        weights_ = np.array([np.roll(row, i) for i, row in enumerate(weights_)])[:, excess_size:-excess_size]
        energies_ = np.array([np.roll(row, i) for i, row in enumerate(energies_)])[:, excess_size:-excess_size]
      else:
        assert rnn.config.typed_value("att_area") == "seg"
        energy_len = out["energy_len"]
        segment_starts = out["segment_starts-out"][0]
        energies_ = np.where(np.arange(0, energies_.shape[1])[None, :] >= energy_len, np.min(energies_), energies_)
        zeros = np.zeros((weights_.shape[0], encoder_len - weights_.shape[1]))  # (dec, enc - win)
        ones = np.ones_like(zeros)
        weights_ = np.concatenate([weights_, zeros], axis=1)  # (dec, enc)
        energies_ = np.concatenate([energies_, ones * np.min(energies_)], axis=1)
        weights_ = np.array([np.roll(row, start) for start, row in zip(segment_starts, weights_)])
        energies_ = np.array([np.roll(row, start) for start, row in zip(segment_starts, energies_)])

      target_align_borders_expanded = [0] + [
        (i + 1) * 6 - 1 if i != len(target_align) - 1 else (i + 1) * 6 - 1 - (6 - last_label_rep) for i, x in
        enumerate(target_align) if x != targetb_blank_idx]
      target_align_borders_expanded_center = [(i + j) / 2 for i, j in
        zip(target_align_borders_expanded[:-1], target_align_borders_expanded[1:])]
      target_label_idxs = [i for i, x in enumerate(target_align) if x != targetb_blank_idx]
      label_ticks_major = [i+1 for i in range(len(target_label_idxs))]
      label_ticks_center = [(i + j) / 2 for i, j in
                            zip(([0] + label_ticks_major)[:-1], ([0] + label_ticks_major)[1:])]
      target_align_major = [x for x in target_align if x != targetb_blank_idx]

      weights_ = np.concatenate(
        [np.repeat(weights_[:, :-1], 6, axis=-1), np.repeat(weights_[:, -1:], last_label_rep, axis=-1)], axis=-1)
      weights_ = weights_[target_label_idxs]

      energies_ = np.concatenate(
        [np.repeat(energies_[:, :-1], 6, axis=-1), np.repeat(energies_[:, -1:], last_label_rep, axis=-1)], axis=-1)
      energies_ = energies_[target_label_idxs]

      vocab = rnn.config.typed_value("vocab")
      if "vocab_file" in vocab:
        with open(vocab["vocab_file"]) as f:
          vocab = f.read()
        vocab = ast.literal_eval(vocab)
        vocab = {v: k for k, v in vocab.items()}
        target_align_major = [vocab[c] for c in target_align_major]

      for ax, matrix, title in zip(axs[:, head] if weights.shape[3] > 1 else axs, [weights_, energies_], ["weights", "energies"]):
        # plot matrix
        matshow = ax.matshow(matrix, aspect="auto", cmap=plt.cm.get_cmap("Blues"))
        # create second x axis for hmm alignment labels and plot same matrix
        hmm_ax = ax.twiny()
        time_ax = ax.twiny()
        # set x and y axis for target alignment axis
        ax.set_title(title + "_head" + str(head), y=1.1)

        ax.set_yticks([x - .5 for x in label_ticks_major])
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.set_yticks([x - .5 for x in label_ticks_center], minor=True)
        ax.set_yticklabels(target_align_major, fontsize=9, minor=True)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.tick_params(axis="y", which="major", length=10)

        ax.set_xticks([x - .5 if x == 0 else x + .5 for x in target_align_borders_expanded])
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.set_xticks([x if i == 0 else x + .5 for i, x in enumerate(target_align_borders_expanded_center)], minor=True)
        ax.set_xticklabels(target_align_major, fontsize=9, minor=True, rotation="vertical")
        ax.tick_params(axis="x", which="minor", length=0)
        ax.tick_params(axis="x", which="major", length=10)
        ax.set_xlabel("RNA BPE Alignment")
        ax.set_ylabel("Output RNA BPE Labels")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.spines['top'].set_position(('outward', 50))
        for tick in ax.xaxis.get_minor_ticks():
          tick.label1.set_horizontalalignment("center")

        # set x ticks and labels and positions for hmm axis
        hmm_ax.set_xticks([x - .5 for x in hmm_align_borders])
        hmm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
        hmm_ax.set_xticks([x - .5 for x in hmm_align_borders_center], minor=True)
        hmm_ax.set_xticklabels(hmm_align_major, fontsize=9, minor=True, rotation="vertical")
        hmm_ax.tick_params(axis="x", which="minor", length=0)
        hmm_ax.tick_params(axis="x", which="major", length=10)
        hmm_ax.xaxis.set_ticks_position('top')
        hmm_ax.xaxis.set_label_position('top')
        hmm_ax.set_xlabel("HMM Phoneme Alignment")

        time_ax.set_xlabel("Input Time Frames")
        time_ax.xaxis.tick_bottom()
        time_ax.xaxis.set_label_position('bottom')
        time_ax.set_xlim(ax.get_xlim())
        time_ax.set_xticks([x - .5 for x in range(0, len(hmm_align), 10)])
        time_ax.set_xticklabels([x for x in range(0, len(hmm_align), 10)])

        for idx in hmm_align_borders:
          ax.axvline(x=idx - .5, ymin=0, ymax=1, color="k", linestyle="--", linewidth=.5)
        for idx in label_ticks_major:
          ax.axhline(y=idx - .5, xmin=0, xmax=1, color="k", linewidth=.5)

        plt.colorbar(matshow, ax=ax)

    suptitle = options.model_name + "\n" + seq_tag[0]
    fig.suptitle(suptitle)
    # fig.tight_layout()
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
    plt.savefig(os.path.join(plot_dir, options.model_name + "_seq-" + str(seq_idx) + ".pdf"))
    plt.close()

    seq_idx += 1

  print("Done. More seqs which we did not dumped: %s" % dataset.is_less_than_num_seqs(seq_idx), file=log.v1)


def init(config_filename, command_line_options):
  """
  :param str config_filename:
  :param list[str] command_line_options:
  """
  rnn.init(config_filename=config_filename, command_line_options=command_line_options, config_updates={"log": None},
    extra_greeting="RETURNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config,
                                    train_data=rnn.train_data)  # rnn.engine.init_network_from_config(rnn.config)


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('returnn_config')
  arg_parser.add_argument('--epoch', type=int, default=1)
  arg_parser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  arg_parser.add_argument('--endseq', type=int, default=10, help='end seq idx (inclusive) or -1 (default: 10)')
  arg_parser.add_argument('--out_dir', type=str)
  arg_parser.add_argument('--model_name', type=str)
  args = arg_parser.parse_args(argv[1:])

  init(config_filename=args.returnn_config, command_line_options=[])
  dump(rnn.train_data, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
