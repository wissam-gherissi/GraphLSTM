import torch, io
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from p_tqdm import p_map


def get_default_metrics(sample_input):
    def metrics_fn(model):
        with profile(activities=[ProfilerActivity.CPU],
                     record_shapes=True, profile_memory=True, with_flops=True) as prof:
            with record_function("model_inference"):
                model(sample_input)
        flops = prof.key_averages().total_average().flops / 1e6
        time = prof.key_averages().self_cpu_time_total / 1e3
        mem = prof.key_averages().total_average().cpu_memory_usage / 1e9
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return [flops, time, mem, params]

    return metrics_fn


def norm(v):
    return (v - v.min()) / (v.max() - v.min())


class Data:
    def __init__(self, encodings, pretrain_set_size=2000):
        self.encodings = torch.Tensor(np.array(encodings))
        self.pr_indices = np.random.default_rng().choice(len(self.encodings),
                                                         size=pretrain_set_size,
                                                         replace=False)
        self.pr_encodings = self.encodings[self.pr_indices]
        self.pr_data = None
        self.ensemble = None
        self.bench_evals = None

    def generate_pretraining_data(self, sample_input, encoding_to_module,
                                  threads=4):
        metrics_fn = get_default_metrics(sample_input)
        enc_to_metrics = lambda enc: metrics_fn(encoding_to_module(enc))
        #res = torch.Tensor(p_map(enc_to_metrics,
        #                         self.pr_encodings,
        #                         num_cpus=1))

        res = []
        for enc in self.pr_encodings:
            res.append(enc_to_metrics(enc))
        res = torch.Tensor(res)
            
        self.pr_data = []
        for i in range(len(res[0])):
            self.pr_data.append(norm(res[:, i]))

    def get_benchmark_metrics(self, metrics_fn,
                              threads=4, from_index=False):
        if from_index:
            inputs = self.pr_indices
        else:
            inputs = self.pr_encodings
        res = torch.Tensor(p_map(metrics_fn,
                                 inputs,
                                 num_cpus=threads))

        self.pr_data = []
        for i in range(len(res[0])):
            self.pr_data.append(norm(res[:, i]))

    def get_benchmark_evals(self, evals_fn,
                            threads=4, from_index=False):
        if from_index:
            inputs = list(range(len(self.encodings)))
        else:
            inputs = self.encodings
        res = torch.Tensor(p_map(evals_fn,
                                 inputs,
                                 num_cpus=threads))
        self.bench_evals = []
        for i in range(len(res[0])):
            self.bench_evals.append(norm(res[:, i]))
