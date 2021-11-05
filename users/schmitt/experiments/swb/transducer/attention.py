from enum import Enum


class AttentionTypes(Enum):
  """The order matters as the functions below use indices for the attention types"""
  NO = 0
  LOCAL_WIN = 1


def add_attention(net_dict, attention_type):
  """This function expects a network dictionary of an "extended transducer" model and adds a self-attention mechanism
  according to some parameters set in the returnn config.

  The parameters are currently set inside the config in order to not modify the header of this function, which would
  lead to a different hash value of the ReturnnConfig object. The currently supported parameters are:

  :param att_type:
  :param att_area:
  :param att_win_size:
  :param att_seg_left_size:
  :param att_seg_right_size:
  :param att_seg_clamp_size:
  :param mask_att:
  :param att_locations:
  :param att_seg_use_emb:

  :param net_dict:
  :param attention_type: int in [0, 1, 2, ...]
  :return: dict net_dict with added attention mechanism
  """

  if att_locations is None:
    net_dict["output"]["unit"]["lm_masked"]["unit"]["subnetwork"]["lstm0"]["from"].append("base:am")
    return

  if mask_att:
    att_name = "att_unmasked"
  else:
    att_name = "att"

  if "slow_rnn" in att_locations:
    net_dict["output"]["unit"]["lm_masked"]["unit"]["subnetwork"]["lstm0"]["from"].append("base:" + att_name)
  if "readout" in att_locations:
    net_dict["output"]["unit"]["readout_in"]["from"].append(att_name)
  if "label_dist" in att_locations:
    net_dict["output"]["unit"]["label_log_prob"]["from"].append(att_name)

  if att_type == "dot":
    # use the dot-product to calculate the energies
    net_dict["output"]["unit"].update({'att_energy': {  # (B, t_att, 1)
      "class": "dot", "red1": "f", "red2": "f", "var1": "spatial:-1", "var2": None,
      "from": ['att_ctx', 'att_query']}})
  elif att_type == "mlp":
    # use an MLP to calculate the energies
    net_dict["output"]["unit"].update({
      'att_energy_in': {  # (B, t_att, D)
        "class": "combine", "kind": "add", "from": ["att_ctx", "att_query"], "n_out": eval("EncKeyTotalDim")},
      "energy_tanh": {
        "class": "activation", "activation": "tanh", "from": ["att_energy_in"]},  # (B, W, D)
      "att_energy": {  # (B, t_att, 1)
        "class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
       })
  # calculate attention weights by applying softmax to the energies
  net_dict["output"]["unit"].update({
    "att_query": {  # (B,D)
      "class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim")},
    'att_weights0': {
      "class": "softmax_over_spatial", "from": 'att_energy', "axis": "spatial:-1",
      "energy_factor": eval("EncKeyPerHeadDim") ** -0.5},
    'att_weights': {
      "class": "dropout", "dropout_noise_shape": {"*": None}, "from": 'att_weights0',
      "dropout": eval("AttentionDropout")},
  })

  # the following attention calculation has two cases, because I wasn't able to implement it using the same code. I
  # will try to fix it in the future. The math behind both cases is the same, just using different layers.
  if att_area == "win" and att_win_size == "full":
    # calculate attention over all encoder frames
    net_dict["output"]["unit"]['att'] = {"class": "generic_attention", "weights": "att_weights", "base": "att_val"}
  else:
    # calculate attention in all other cases
    net_dict["output"]["unit"].update({
      'att0': {"class": "dot", "from": ["att_val", 'att_weights'], "red1": "spatial:-1", "red2": -1, "var1": "f",
               "var2": "static:0"},  # (B, 1, V)
      "att": {"class": "merge_dims", "from": "att0", "axes": "except_time"}  # (B,V)
    })

  if att_area == "win":
    """Local window attention"""
    if type(att_win_size) == int:
      # In this case, the local window has a given fixed size

      # extract the window inside the encoder (more efficient than doing in inside the decoder)
      net_dict.update({
        "enc_ctx0": {
          "class": "linear", "from": "encoder", "activation": None, "with_bias": False,
          "n_out": eval("EncKeyTotalDim"), "L2": eval("l2"), "dropout": 0.2},  # (B,T,D)
        "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": att_win_size},  # [B,T,W,D]
        "enc_val": {"class": "copy", "from": "encoder"},  # (B,T,V)
        "enc_val_win": {"class": "window", "from": "enc_val", "window_size": att_win_size},  # [B,T,W,V]
      })
      # context and value are just the windows at the corresponding step in the decoder
      net_dict["output"]["unit"].update(
        {"att_ctx": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
        "att_val": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,V],
      })
    elif att_win_size == "full":
      # in this case, we calculate the attention over the whole encoder sequence
      net_dict["output"]["unit"].update({
        "att_ctx0": {  # (B, T, D)
          "class": "linear", "from": "base:encoder", "activation": None, "with_bias": False,
          "n_out": eval("EncKeyTotalDim"), "L2": eval("l2"), "dropout": 0.2},
        "att_ctx": {
          "class": "reinterpret_data", "from": "att_ctx0", "set_dim_tags": {
            "t": DimensionTag(kind=DimensionTag.Types.Spatial, description="att_t")}},
        "att_val": {  # (B,T,V)
          "class": "reinterpret_data", "from": "base:encoder", "set_dim_tags": {
            "t": DimensionTag(kind=DimensionTag.Types.Spatial, description="att_t")}},
      })
    else:
      raise ValueError("att_win_size needs to be an integer or 'full'")

  else:
    """Segmental Attention: a segment is defined as current frame + all previous blank frames"""
    assert att_area == "seg"
    # add the base attention mechanism here. The variations below define the segment boundaries (segment_starts and
    # segment_lens)
    net_dict["output"]["unit"].update({
      "const1": {"class": "constant", "value": 1},
      "segments": {  # [B,t_sliced,D]
        "class": "slice_nd", "from": "base:encoder", "start": "segment_starts", "size": "segment_lens"},
      "att_ctx0": {"class": "copy", "from": "segments"},
      "att_ctx": {  # [B,D]
        "class": "linear", "from": "att_ctx0", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim"),
        "L2": eval("l2"), "dropout": 0.2},
      "att_val": {"class": "copy", "from": "segments"}
    })

    # define segment as all frames since the last non-blank output
    # the last non-blank frame is excluded; the current frame is included
    net_dict["output"]["unit"].update({
      "segment_starts": {  # (B,)
        "class": "switch", "condition": "prev:output_is_not_blank", "true_from": ":i",
        "false_from": "prev:segment_starts", "initial_output": 0},
      "segment_lens0": {"class": "combine", "kind": "sub", "from": [":i", "segment_starts"]},
      "segment_lens": {"class": "combine", "kind": "add", "from": ["segment_lens0", "const1"]},  # (B,)
    })

    if att_seg_clamp_size is not None:
      # in this case, we clamp all segments to the specified size
      # this might make sense since the first segment is often much longer than the other segments
      assert type(att_seg_clamp_size) == int
      net_dict["output"]["unit"]["segment_starts0"] = net_dict["output"]["unit"]["segment_starts"].copy()
      net_dict["output"]["unit"]["segment_starts0"]["false_from"] = "prev:segment_starts0"
      net_dict["output"]["unit"]["segment_lens0_copy"] = net_dict["output"]["unit"]["segment_lens0"].copy()
      net_dict["output"]["unit"]["segment_lens0_copy"]["from"] = [":i", "segment_starts0"]
      net_dict["output"]["unit"]["segment_lens1_copy"] = net_dict["output"]["unit"]["segment_lens"].copy()
      net_dict["output"]["unit"]["segment_lens1_copy"]["from"] = ["segment_lens0_copy", "const1"]
      net_dict["output"]["unit"].update({
        "clamp_size": {"class": "constant", "value": att_seg_clamp_size},
        "clamp_mask": {"class": "compare", "from": ["segment_lens1_copy", "clamp_size"], "kind": "greater"},
        "clamped_diff": {"class": "combine", "from": ["segment_lens1_copy", "clamp_size"], "kind": "sub"},
        "clamped_start": {"class": "combine", "from": ["segment_starts0", "clamped_diff"], "kind": "add"},
        "segment_starts": {"class": "switch", "condition": "clamp_mask", "true_from": "clamped_start",
          "false_from": "segment_starts0"},
        })

    if att_seg_left_size is not None:
      # in this case, we add a specified number of frames on the left side of the segment
      if att_seg_clamp_size is None:
        idx = 0
      else:
        idx = 1
      net_dict["output"]["unit"]["const_left_win_size"] = {"class": "constant", "value": att_seg_left_size}
      net_dict["output"]["unit"]["segment_starts" + str(idx)] = net_dict["output"]["unit"]["segment_starts"].copy()
      if att_seg_clamp_size is None:
        net_dict["output"]["unit"]["segment_starts" + str(idx)]["false_from"] = "prev:segment_starts" + str(idx)
      net_dict["output"]["unit"].update({
        "segment_starts" + str(idx + 1): {
          "class": "combine", "from": ["segment_starts" + str(idx), "const_left_win_size"], "kind": "sub"},
        "less_than_0": {"class": "compare", "from": "segment_starts" + str(idx+1), "value": 0, "kind": "less"},
        "segment_starts": {
          "class": "switch", "condition": "less_than_0", "true_from": 0,
          "false_from": "segment_starts" + str(idx+1)}, })

    if att_seg_right_size is not None:
      # in this case, we add a specified number of frames on the right side of the segment
      net_dict["output"]["unit"]["segment_lens1"] = net_dict["output"]["unit"]["segment_lens"].copy()
      net_dict["output"]["unit"]["const_right_win_size"] = {"class": "constant", "value": att_seg_right_size}
      net_dict["output"]["unit"].update({
        "segment_lens2": {
          "class": "combine", "kind": "add", "from": ["segment_lens1", "const_right_win_size"]},
        "seq_lens": {"class": "length", "from": "base:encoder"},
        "max_length": {"class": "combine", "from": ["seq_lens", "segment_starts"], "kind": "sub"},
        "last_seg_idx": {"class": "combine", "from": ["segment_starts", "segment_lens1"], "kind": "add"},
        "greater_than_length": {"class": "compare", "from": ["last_seg_idx", "seq_lens"], "kind": "greater_equal"},
        "segment_lens": {
          "class": "switch", "condition": "greater_than_length", "true_from": "max_length",
          "false_from": "segment_lens2"},
         })

    if mask_att is True:
      # in this case, we create a layer "att_unmasked" which is the attention vector over the previous segment
      net_dict["output"]["unit"].update({
        "att_masked": {
          "class": "masked_computation", "mask": "prev:output_is_not_blank", "from": "prev:att",
          "unit": {"class": "copy", "from": "data"}},
        "att_unmasked": {"class": "unmask", "from": "att_masked", "mask": "prev:output_is_not_blank"}, })

    if att_seg_use_emb and (att_seg_left_size or att_seg_right_size):
      # in this case, we add an one-hot embedding to the encoder frames, indicating whether they belong to the
      # current segment or not
      if att_seg_left_size:
        net_dict["output"]["unit"]["segment_left_index"] = {
          "class": "combine", "from": ["segment_starts0", "segment_starts"], "kind": "sub"
        }
        net_dict["output"]["unit"]["real_seg_len"] = {"class": "combine",
          "from": [":i", "segment_starts0"], "kind": "sub"}
      else:
        net_dict["output"]["unit"]["segment_left_index"] = {"class": "constant", "value": 0}
        net_dict["output"]["unit"]["real_seg_len"] = {"class": "combine", "from": [":i", "segment_starts"],
                                                      "kind": "sub"}
      net_dict["output"]["unit"]["segment_right_index"] = {"class": "combine",
        "from": ["segment_left_index", "real_seg_len"], "kind": "add"}
      net_dict["output"]["unit"]["segments0"] = net_dict["output"]["unit"]["segments"].copy()
      net_dict["output"]["unit"].update({
        "segment_indices": {"class": "range_in_axis", "from": "segments0", "axis": "dyn:-1", "is_output_layer": True},
        "is_in_segment": {"class": "compare", "from": ["segment_left_index", "segment_indices", "segment_right_index"],
                          "kind": "less_equal"},
        "embedding0": {"class": "switch", "condition": "is_in_segment", "true_from": 0.0, "false_from": 1.0},
        "embedding": {"class": "expand_dims", "from": "embedding0", "axis": -1},
          "embedding_rev0": {"class": "switch", "condition": "is_in_segment", "true_from": 1.0, "false_from": 0.0},
          "embedding_rev": {"class": "expand_dims", "from": "embedding_rev0", "axis": -1},
          "segments": {"class": "copy", "from": ["segments0", "embedding", "embedding_rev"], "is_output_layer": True}})