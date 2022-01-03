
legacy_network = {
    'bwd_lstm_1': {'L2': 0.01, 'class': 'rec', 'direction': -1, 'dropout': 0.1, 'from': 'source', 'n_out': 512, 'unit': 'nativelstm2'},
    'bwd_lstm_2': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': -1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_1', 'bwd_lstm_1'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'bwd_lstm_3': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': -1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_2', 'bwd_lstm_2'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'bwd_lstm_4': {'L2': 0.01, 'class': 'rec', 'direction': -1, 'dropout': 0.1, 'from': 'max_pool_3', 'n_out': 512, 'unit': 'nativelstm2'},
    'bwd_lstm_5': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': -1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_4', 'bwd_lstm_4'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'bwd_lstm_6': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': -1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_5', 'bwd_lstm_5'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'fwd_lstm_1': {'L2': 0.01, 'class': 'rec', 'direction': 1, 'dropout': 0.1, 'from': 'source', 'n_out': 512, 'unit': 'nativelstm2'},
    'fwd_lstm_2': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': 1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_1', 'bwd_lstm_1'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'fwd_lstm_3': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': 1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_2', 'bwd_lstm_2'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'fwd_lstm_4': {'L2': 0.01, 'class': 'rec', 'direction': 1, 'dropout': 0.1, 'from': 'max_pool_3', 'n_out': 512, 'unit': 'nativelstm2'},
    'fwd_lstm_5': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': 1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_4', 'bwd_lstm_4'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'fwd_lstm_6': { 'L2': 0.01,
                    'class': 'rec',
                    'direction': 1,
                    'dropout': 0.1,
                    'from': ['fwd_lstm_5', 'bwd_lstm_5'],
                    'n_out': 512,
                    'unit': 'nativelstm2'},
    'max_pool_3': {'class': 'pool', 'from': ['fwd_lstm_3', 'bwd_lstm_3'], 'mode': 'max', 'padding': 'same', 'pool_size': (2,), 'trainable': False},
    'output': { 'class': 'softmax',
                'from': ['fwd_lstm_6', 'bwd_lstm_6'],
                'loss': 'fast_bw',
                'loss_opts': { 'sprint_opts': { 'minPythonControlVersion': 4,
                                                'numInstances': 2,
                                                'sprintConfigStr': '--config=rasr.loss.config --*.LOGFILE=nn-trainer.loss.log --*.TASK=1',
                                                'sprintExecPath': '/u/rossenbach/src/rasr_wei/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard',
                                                'usePythonSegmentOrder': False},
                               'tdp_scale': 0.0},
                'n_out': 139,
                'target': None},
    #'source': {'class': 'eval', 'eval': "self.network.get_config().typed_value('_specaugment_eval_func')(source(0, as_data=True), network=self.network)"}
    'source': {'class': 'copy', 'from': ["data"]}
}

from returnn_common.nn import Module, LayerRef, get_extern_data, get_root_extern_data, NameCtx, make_root_net_dict
from returnn_common import nn

from .specaugment_clean_v2 import specaugment, SpecAugmentSettings

class BLSTMPoolModule(Module):

    def __init__(self, hidden_size, pool, dropout=None, l2=None):
        super().__init__()
        self.fw_rec = nn.LSTM(n_out=hidden_size, direction=1, l2=l2)
        self.bw_rec = nn.LSTM(n_out=hidden_size, direction=-1, l2=l2)
        self.pool = pool
        self.dropout = dropout

    def forward(self, inp):
        fw_out, _ = self.fw_rec(inp)
        bw_out, _ = self.bw_rec(inp)
        concat = nn.concat((fw_out, "F"), (bw_out, "F"))
        if self.pool is not None and self.pool > 1:
            inp = nn.dropout(nn.pool(concat, mode="max", pool_size=(self.pool,), padding="same"), self.dropout)
        else:
            inp = nn.dropout(concat, self.dropout)
        return inp


class BLSTMCTCModel(Module):

    def __init__(self, num_nn, size, max_pool, num_labels, dropout=None, l2=None, specaugment_settings=None):
        """

        :param num_nn:
        :param size:
        :param list[int] max_pool:
        :param dropout:
        :param SpecAugmentSettings specaugment_settings:
        """
        super().__init__()

        self.specaugment_settings = specaugment_settings

        modules = []
        for i in range(num_nn - 1):
            pool = max_pool[i] if i < len(max_pool) else 1
            modules.append(BLSTMPoolModule(size, pool, dropout=dropout, l2=l2))
        last_pool = max_pool[-1] if len(max_pool) == num_nn else 1
        self.last_blstm = BLSTMPoolModule(size, last_pool, dropout=dropout, l2=l2)
        self.blstms = nn.Sequential(modules)
        self.linear = nn.Linear(n_out=num_labels, with_bias=True)

    def forward(self):
        inp = get_root_extern_data("data")
        if self.specaugment_settings:
            inp = specaugment(inp, **self.specaugment_settings.get_options())
        inp = self.blstms(inp)
        inp = self.last_blstm(inp)
        out = self.linear(inp)
        out = nn.softmax(out, name="output", axis="F")
        return out


def get_network(*args, **kwargs):
    blstm_ctc = BLSTMCTCModel(*args, **kwargs)
    net_dict = make_root_net_dict(blstm_ctc)

    return net_dict

