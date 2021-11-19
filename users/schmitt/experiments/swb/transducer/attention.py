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
  net_dict["output"]["unit"].update({
    "att_masked": {
      "class": "masked_computation", "mask": "prev:output_is_not_blank", "from": "prev:att",
      "unit": {"class": "copy", "from": "data"}},
    "att_unmasked": {"class": "unmask", "from": "att_masked", "mask": "prev:output_is_not_blank"}, })

  if att_type == "dot":
    # use the dot-product to calculate the energies
    net_dict["output"]["unit"].update({'att_energy': {  # (B, t_att, 1)
      "class": "dot", "red1": "f", "red2": "f", "var1": "spatial:-1", "var2": None,
      "from": ['att_ctx', 'att_query']}})
  elif att_type == "mlp":
    # use an MLP to calculate the energies
    net_dict["output"]["unit"].update({
      'att_energy_in': {  # (B, t_att, D)
        "class": "combine", "kind": "add",
        "from": ["att_ctx", "att_query"],
        "n_out": eval("EncKeyTotalDim")},
      "energy_tanh": {
        "class": "activation", "activation": "tanh", "from": ["att_energy_in"]},  # (B, W, D)
      "att_energy": {  # (B, t_att, 1)
        "class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
       })
  # calculate attention weights by applying softmax to the energies
  net_dict["output"]["unit"].update({
    "att_query": {  # (B,D)
      "class": "linear", "from": att_query_in, "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim")},
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
    net_dict["output"]["unit"]['att'] = {"class": "generic_attention", "weights": "att_weights",
                                         "base": "att_val"}
  else:
    # calculate attention in all other cases
    net_dict["output"]["unit"].update({
      'att0': {"class": "dot", "from": ["att_val", 'att_weights'], "red1": "spatial:-1",
              "red2": "static:-1" if att_area == "win" else "dyn:-1",
               "var1": "f", "var2": "static:0"},  # (B, 1, V)
      "att": {"class": "merge_dims", "from": "att0", "axes": "except_time"}  # (B,V)
    })

  # define segment as all frames since the last non-blank output
  # the last non-blank frame is excluded; the current frame is included
  net_dict["output"]["unit"].update({
    "const1": {"class": "constant", "value": 1},
    "const0.0_0": {"class": "constant", "value": 0.0, "with_batch_dim": True},
    "const0.0": {"class": "expand_dims", "axis": "F", "from": "const0.0_0"},
    "const1.0_0": {"class": "constant", "value": 1.0, "with_batch_dim": True},
    "const1.0": {"class": "expand_dims", "axis": "F", "from": "const1.0_0"},
    "segment_starts": {  # (B,)
      "class": "switch", "condition": "prev:output_is_not_blank", "true_from": ":i",
      "false_from": "prev:segment_starts", "initial_output": 0},
    "segment_lens0": {"class": "combine", "kind": "sub", "from": [":i", "segment_starts"]},
    "segment_lens": {"class": "combine", "kind": "add", "from": ["segment_lens0", "const1"]},  # (B,)
  })

  if att_seg_use_emb and att_seg_emb_size:
    def get_one_hot_embedding(nd, idx):
      return {"class": "copy", "from": ["const0.0" if i != idx else "const1.0" for i in range(nd)]}

    if att_seg_emb_size >= 2:
      conditions = {
        "is_in_segment": {
          "class": "compare", "from": ["segment_left_index", "segment_indices", "segment_right_index"],
          "kind": "less_equal"}, }
    if att_seg_emb_size >= 3:
      conditions.update({
        "left_of_segment": {
          "class": "compare", "from": ["segment_left_index", "segment_indices"], "kind": "greater"}})
    if att_seg_emb_size == 4:
      conditions = dict({
        "is_cur_step": {
          "class": "compare", "from": ["segment_indices", "segment_right_index"], "kind": "equal"}, **conditions})

    net_dict["output"]["unit"].update(conditions)
    for i in range(att_seg_emb_size):
      net_dict["output"]["unit"]["emb" + str(i)] = get_one_hot_embedding(att_seg_emb_size, i)

    # TODO: this condition is due to some legacy models which still need to do search
    # the legacy model used 2D embedding and used [0,1] in case that the frame was in the segment
    # this new model uses [1,0]
    if att_seg_emb_size > 2:
      for i, cond in enumerate(conditions):
        if i == len(conditions) - 1:
          net_dict["output"]["unit"].update({
            "embedding" + str(i): {
              "class": "switch", "condition": cond, "true_from": "emb" + str(i), "false_from": "emb" + str(i + 1)}, })
        else:
          net_dict["output"]["unit"].update({
            "embedding" + str(i): {
              "class": "switch", "condition": cond, "true_from": "emb" + str(i),
              "false_from": "embedding" + str(i + 1)}, })
    else:
      net_dict["output"]["unit"].update({
        "embedding0": {
          "class": "switch", "condition": "is_in_segment", "true_from": "emb1", "false_from": "emb0"}, })

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
      if att_weight_feedback and att_type == "mlp":
        net_dict.update({
          "inv_fertility": {
            "class": "linear", "activation": "sigmoid", "with_bias": False, "from": "encoder", "n_out": 1}, })
        net_dict["output"]["unit"].update({
          "weight_feedback": {
            "class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
            "n_out": eval("EncKeyTotalDim")},
          "accum_att_weights": {
            "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
            "eval": "source(0) + source(1) * source(2) * 0.5",
            "out_type": {"dim": 1, "shape": (None, 1)}},
        })
        net_dict["output"]["unit"]["att_energy_in"]["from"].append("weight_feedback")

        if not (att_seg_use_emb and att_seg_emb_size):
          net_dict.update({
            "enc_ctx": {  # (B, T, D)
              "class": "linear", "from": "encoder", "activation": None, "with_bias": False,
              "n_out": eval("EncKeyTotalDim"), "L2": eval("l2"), "dropout": 0.2},
            "enc_val": {"class": "copy", "from": ["encoder"]},
          })
          net_dict["output"]["unit"]["att_energy_in"]["from"] = ["base:enc_ctx" if item == "att_ctx" else item for item
                                                                 in net_dict["output"]["unit"]["att_energy_in"]["from"]]
          net_dict["output"]["unit"]['att']["base"] = "base:enc_val"
        else:
          net_dict.update({
            "segment_indices": {"class": "range_in_axis", "from": "encoder", "axis": "t"},
          })

          net_dict["output"]["unit"].update({
            "segment_left_index": {"class": "copy", "from": ["segment_starts"]},
            "segment_right_index": {"class": "combine", "from": ["segment_starts", "segment_lens0"], "kind": "add"},
            "att_val": {"class": "copy", "from": ["base:encoder", "embedding0"]},
            "att_ctx": {  # (B, T, D)
              "class": "linear", "from": ["base:encoder", "embedding0"], "activation": None, "with_bias": False,
              "n_out": eval("EncKeyTotalDim"), "L2": eval("l2"), "dropout": 0.2}
          })
          for cond in ["is_in_segment", "left_of_segment", "is_cur_step"]:
            if cond in net_dict["output"]["unit"]:
              net_dict["output"]["unit"][cond]["from"] = ["base:segment_indices"
                                                          if item == "segment_indices" else item
                                                          for item in net_dict["output"]["unit"][cond]["from"]]

      else:
        key_time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="att_t")
        net_dict["output"]["unit"].update({
          "att_ctx0": {  # (B, T, D)
            "class": "linear", "from": ["base:encoder"], "activation": None, "with_bias": False,
            "n_out": eval("EncKeyTotalDim"), "L2": eval("l2"), "dropout": 0.2}, "att_ctx": {
            "class": "reinterpret_data", "from": ["att_ctx0"], "set_dim_tags": {
              "T": key_time_tag}},
          "att_val": {  # (B,T,V)
            "class": "reinterpret_data", "from": ["base:encoder"], "set_dim_tags": {
              "T": key_time_tag}}, })

        if att_seg_use_emb and att_seg_emb_size:
          net_dict["output"]["unit"]["att_val0"] = net_dict["output"]["unit"]["att_val"].copy()
          net_dict["output"]["unit"]["att_ctx"] = net_dict["output"]["unit"]["att_ctx0"].copy()
          net_dict["output"]["unit"]["att_ctx"]["from"] = "att_val"
          net_dict["output"]["unit"].update({
            "segment_indices": {"class": "range_in_axis", "from": "att_val0", "axis": "t"},
            "segment_left_index": {"class": "copy", "from": ["segment_starts"]},
            "segment_right_index": {"class": "combine", "from": ["segment_starts", "segment_lens0"], "kind": "add"},
            "att_val1": {"class": "copy", "from": ["att_val0", "embedding0"]},
            "att_val": {  # (B,T,V)
              "class": "reinterpret_data", "from": ["att_val1"], "set_axes": {
                "t": "stag:att_t"}},
          })


    else:
      raise ValueError("att_win_size needs to be an integer or 'full'")

  else:
    """Segmental Attention: a segment is defined as current frame + all previous blank frames"""
    assert att_area == "seg"
    # add the base attention mechanism here. The variations below define the segment boundaries (segment_starts and
    # segment_lens)
    net_dict["output"]["unit"].update({
      "segments": {  # [B,t_sliced,D]
        "class": "slice_nd", "from": "base:encoder", "start": "segment_starts", "size": "segment_lens"},
      "att_ctx0": {"class": "copy", "from": "segments"},
      "att_ctx": {  # [B,D]
        "class": "linear", "from": "att_ctx0", "activation": None, "with_bias": False, "n_out": eval("EncKeyTotalDim"),
        "L2": eval("l2"), "dropout": 0.2},
      "att_val": {"class": "copy", "from": "segments"}
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
      net_dict["output"]["unit"]["segment_starts" + str(idx)] = net_dict["output"]["unit"]["segment_starts"].copy()
      if att_seg_clamp_size is None:
        net_dict["output"]["unit"]["segment_starts" + str(idx)]["false_from"] = "prev:segment_starts" + str(idx)

      if type(att_seg_left_size) == int:
        net_dict["output"]["unit"].update({
          "const_left_win_size": {"class": "constant", "value": att_seg_left_size},
        })
      elif att_seg_left_size == "full":
        net_dict["output"]["unit"].update({
          "const_left_win_size": {"class": "copy", "from": "segment_starts" + str(idx)}, })
      net_dict["output"]["unit"].update({
        "segment_starts" + str(idx + 1): {
          "class": "combine", "from": ["segment_starts" + str(idx), "const_left_win_size"], "kind": "sub"},
        "less_than_0": {"class": "compare", "from": "segment_starts" + str(idx + 1), "value": 0, "kind": "less"},
        "segment_starts": {
          "class": "switch", "condition": "less_than_0", "true_from": 0,
          "false_from": "segment_starts" + str(idx + 1)}, })

    if att_seg_right_size is not None:
      # in this case, we add a specified number of frames on the right side of the segment
      net_dict["output"]["unit"]["segment_lens1"] = net_dict["output"]["unit"]["segment_lens"].copy()
      net_dict["output"]["unit"]["seq_lens"] = {"class": "length", "from": "base:encoder"}
      if type(att_seg_right_size) == int:
        net_dict["output"]["unit"].update({
          "const_right_win_size": {"class": "constant", "value": att_seg_right_size},
          "segment_lens2": {
            "class": "combine", "kind": "add", "from": ["segment_lens1", "const_right_win_size"]},
          "max_length": {"class": "combine", "from": ["seq_lens", "segment_starts"], "kind": "sub"},
          "last_seg_idx": {"class": "combine", "from": ["segment_starts", "segment_lens2"], "kind": "add"},
          "greater_than_length": {"class": "compare", "from": ["last_seg_idx", "seq_lens"], "kind": "greater_equal"},
          "segment_lens": {
            "class": "switch", "condition": "greater_than_length", "true_from": "max_length",
            "false_from": "segment_lens2"}, })
      elif att_seg_right_size == "full":
        net_dict["output"]["unit"].update({
          "segment_lens": {
            "class": "combine", "from": ["seq_lens", "segment_starts"], "kind": "sub"},
           })

    if att_seg_use_emb and att_seg_emb_size and (att_seg_left_size or att_seg_right_size):
      # in this case, we add an one-hot embedding to the encoder frames, indicating whether they belong to the
      # current segment or not
      if att_seg_left_size:
        # first in-segment-index of slice_nd output
        net_dict["output"]["unit"]["segment_left_index"] = {
          "class": "combine", "from": ["segment_starts0", "segment_starts"], "kind": "sub"
        }
        # the length of the segment (without additional window)
        net_dict["output"]["unit"]["real_seg_len"] = {"class": "combine",
          "from": [":i", "segment_starts0"], "kind": "sub"}
      else:
        # first in-segment-index of slice_nd output
        net_dict["output"]["unit"]["segment_left_index"] = {"class": "constant", "value": 0}
        # the length of the segment (without additional window)
        net_dict["output"]["unit"]["real_seg_len"] = {"class": "combine", "from": [":i", "segment_starts"],
                                                      "kind": "sub"}
      # last in-segment-index of slice_nd output
      net_dict["output"]["unit"]["segment_right_index"] = {"class": "combine",
        "from": ["segment_left_index", "real_seg_len"], "kind": "add"}

      # 'segments' is redefined here, therefore we need to rename the existing layer
      net_dict["output"]["unit"]["segments0"] = net_dict["output"]["unit"]["segments"].copy()

      net_dict["output"]["unit"].update({
        "segment_indices": {"class": "range_in_axis", "from": "segments0", "axis": "dyn:-1"},
        "segments": {"class": "copy", "from": ["segments0", "embedding0"], "is_output_layer": True},

      })
