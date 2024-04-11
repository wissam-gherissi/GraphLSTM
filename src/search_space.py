import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from p_tqdm import p_map
from os.path import isfile
import dill

from .data import Data
from .ensemble import Ensemble
from .search import SearchMF


# SearchSpace
# encodings
# encoding_to_net
# eval_one_net
# generate_data
# pretrain_ensemble
# launch_search
# save
# load
# current stats ?


class EncoderModule(nn.Module):
    def __init__(self, encoding_dim, n_objectives):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(encoding_dim, 64), nn.LayerNorm(64),
                                    nn.SiLU(),
                                    nn.Linear(64, 128), nn.LayerNorm(128),
                                    nn.SiLU(),
                                    nn.Linear(128, 256), nn.LayerNorm(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256), nn.LayerNorm(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256), nn.LayerNorm(256),
                                    nn.SiLU())
        self.specialized_list = nn.ModuleList([nn.Sequential(nn.Linear(256, 512), nn.LayerNorm(512),
                                                             nn.SiLU()) for _ in range(n_objectives)])
        self.embedding_dim = 512

def encoder_generator_func(encoding_dim, n_objectives):
    return EncoderModule(encoding_dim, n_objectives)


class CachedEvalsTensor:
    def __init__(self, input_tensor, eval_function):
        self.input_tensor = input_tensor
        self.evals = torch.full((len(input_tensor),), -1.)
        self.eval_function = eval_function

    def __getitem__(self, idx):
        if type(idx) == int:
            idx_ = torch.zeros(len(self.input_tensor))
            idx_[idx] = 1
            idx = idx_
        to_be_evaluated = torch.logical_and(idx, self.evals == -1)
        self.evals[to_be_evaluated] = torch.Tensor(self.eval_function(self.input_tensor[to_be_evaluated]))
        return self.evals[idx == 1]

    
def create_search_space(name, save_filename, encodings,
                        encoding_to_net,
                        generate_ensemble_encoder=None,
                        device='cpu', cpu_threads=6,):
    return SearchSpace(name, save_filename, encodings, encoding_to_net,
                       generate_ensemble_encoder, device, cpu_threads)
                                

class SearchSpace():
    def __init__(self, name, save_filename, encodings,
                 encoding_to_net,
                 generate_ensemble_encoder=None,
                 device='cpu', cpu_threads=6):
        self.name = name
        self.encodings = torch.Tensor(encodings)
        if generate_ensemble_encoder is None:
            def generate_ensemble_encoder():
                return encoder_generator_func(encoding_dim=len(encodings[0]), n_objectives=2)
            self.generate_ensemble_encoder = generate_ensemble_encoder
            self.embedding_dim = self.generate_ensemble_encoder().embedding_dim
        self.device = device
        self.cpu_threads = cpu_threads
        self.save_filename = save_filename
        self.encoding_to_net = encoding_to_net
        self.ensemble = None
        self.data = None
        self.search = None

    def preprocess(self, sample_input=None,
                   pretrain_epochs=500,
                   pretrain_set_size=2000,
                   threads=4,
                   custom_data_object=None):
        assert (sample_input is not None or custom_data_object is not None), "You have to provide a sample" \
                                                                             "input unless you're using a" \
                                                                             "custom_data_object " \
                                                                             "(e.g. for a benchmark)"
        # generate pretraining data
        if custom_data_object is None:
            self.data = Data(self.encodings, pretrain_set_size)
            print('Generating pretraining data')
            self.data.generate_pretraining_data(sample_input=sample_input,
                                                encoding_to_module=self.encoding_to_net,
                                                threads=threads)
        else:
            self.data = custom_data_object
            self.data.name = self.name

        # instantiate ensemble
        self.init_ensemble(self.data.pr_encodings, self.data.pr_data)

        # pretrain ensemble
        self.ensemble.pretrain()

        self.save()

    def init_ensemble(self, pr_encodings, pr_data, pretrain_epochs):
        self.ensemble = Ensemble(pretrain_configs=pr_encodings,
                                 pretrain_metrics=pr_data,
                                 network_generator_func=self.generate_ensemble_encoder,
                                 embedding_dim=self.embedding_dim,
                                 n_objectives=2,
                                 n_networks=6,
                                 accelerator=self.device,
                                 devices=self.cpu_threads,
                                 train_lr=5e-3,
                                 pretrain_epochs=pretrain_epochs,
                                 pretrain_lr=1e-2,
                                 pretrain_bs=16)

    def preprocess_no_pretraining(self):
        self.init_ensemble([], [], 0)
        self.save()
        
    def save(self):
        with open(self.save_filename, 'wb') as f:
            dill.dump(self, f)

class SearchInstance():
    
    def __init__(self, name, save_filename, search_space_filename,
                 hi_fi_eval, hi_fi_cost,
                 lo_fi_eval, lo_fi_cost,
                 full_evals_update_freq=10, n_full_evals_per_update=3,
                 part_evals_update_freq=1, n_part_evals_per_update=5,
                 full_epochs_per_iter=30, part_epochs_per_iter=1,
                 device='cpu', threads=6):
        self.name = name
        self.save_filename = save_filename
        with open(search_space_filename, 'rb') as f:
            self.search_space = dill.load(f)
        self.hi_evals = CachedEvalsTensor(self.search_space.encodings, hi_fi_eval)
        self.lo_evals = CachedEvalsTensor(self.search_space.encodings, lo_fi_eval)
        self.search_space.ensemble.accel_init(device)
        self.search = SearchMF(experiment_name=self.name,
                               ensemble=self.search_space.ensemble,
                               all_configs=self.search_space.encodings,
                               all_full_evals=self.hi_evals,
                               full_evals_update_freq=full_evals_update_freq,
                               n_full_evals_per_update=n_full_evals_per_update,
                               full_epochs_per_iter=full_epochs_per_iter,
                               all_part_evals=self.lo_evals,
                               part_evals_update_freq=part_evals_update_freq,
                               n_part_evals_per_update=n_part_evals_per_update,
                               part_epochs_per_iter=part_epochs_per_iter,
                               cost_function=lambda full, part:\
                                     (full * hi_fi_cost + part * lo_fi_cost) / hi_fi_cost)
        self.logs = []
        self.cost_per_iter = n_part_evals_per_update * lo_fi_cost / part_evals_update_freq
        self.cost_per_iter += n_full_evals_per_update * hi_fi_cost / full_evals_update_freq
        self.hi_fi_cost = hi_fi_cost
        
        self.current_iteration = 0
        
    def save(self):
        with open(self.save_filename, 'wb') as f:
            dill.dump(self, f)

    def run_search(self, eval_budget, save_freq=1):
        iterations = int(np.ceil(eval_budget * self.hi_fi_cost // self.cost_per_iter))
        starting_iter = self.current_iteration+1
        for it in range(starting_iter, iterations+1):
            self.logs = self.search.one_loop(it)
            self.current_iteration = it
            if it % save_freq == 0 or it == iterations:
                self.save()
        print(f"\n\n\nTotal Cost: {self.logs[-1]['cost']}\n",
              f"Best solution found:\n",
              f"\t\tVal: {self.logs[-1]['best_val']}\n",
              f"\t\tCfg: {self.logs[-1]['best_solution']}",
              sep='')