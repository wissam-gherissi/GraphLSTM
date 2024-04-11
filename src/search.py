import pickle
from scipy import stats
from os import mkdir
from os.path import exists
import torch


class SearchMF:
    def __init__(self, experiment_name, ensemble, all_configs,
                 all_full_evals, full_evals_update_freq, n_full_evals_per_update, full_epochs_per_iter,
                 all_part_evals, part_evals_update_freq, n_part_evals_per_update, part_epochs_per_iter,
                 cost_function):
        self.ensemble = ensemble
        self.experiment_name = experiment_name

        self.logs = []
        self.best_so_far = torch.Tensor((0.0,0.0,))
        self.best_so_far_cfg = torch.Tensor((0.0,))

        self.all_configs = all_configs
        self.full_evals, self.part_evals = all_full_evals, all_part_evals
        self.full_freq, self.part_freq = full_evals_update_freq, part_evals_update_freq
        self.n_full_evals_per_update, self.n_part_evals_per_update = n_full_evals_per_update, n_part_evals_per_update
        self.full_evals_mask = torch.zeros(all_configs.shape[0], dtype=int)
        self.part_evals_mask = torch.zeros(all_configs.shape[0], dtype=int)

        self.full_epochs_per_iter = full_epochs_per_iter
        self.part_epochs_per_iter = part_epochs_per_iter

        self.cost_fn = cost_function
        self.current_cost = 0.0

    def one_loop(self, it):
        # BO Loop
        
        # Partial evals updates
        if self.part_freq > 0 and it % self.part_freq == 0:
            self.max_acq_function(1)
            self.ensemble.train_multiple(input_data=self.all_configs[self.part_evals_mask==1],
                                         target_data=[torch.Tensor([0] * self.part_evals_mask.sum()),
                                                      self.part_evals[self.part_evals_mask == 1]],
                                         epochs=self.part_epochs_per_iter,
                                         obj_lst=[1])
        # Full evals updates (mask)
        if it % self.full_freq == 0:
            self.max_acq_function(0)
            self.ensemble.train_multiple(input_data=self.all_configs[self.full_evals_mask==1],
                                         target_data=[self.full_evals[self.full_evals_mask == 1],
                                                      torch.Tensor([0] * self.full_evals_mask.sum())],
                                         epochs=self.full_epochs_per_iter,
                                         obj_lst=[0])

        # Log
        self.logs.append(self.evaluate_current_performance())
        
        print(f'\rIt: {it}\tFull evals: {(self.full_evals_mask == 1).sum()}',
              f'Part evals: {(self.part_evals_mask == 1).sum()}',
              f'Equiv. evals: {self.current_cost}',
              f'Best: {self.best_so_far[0].numpy()}',
              sep='\t\t',
              end='\r')
        
        return self.logs
    
    def run(self, iterations):
        for it in range(1, iterations+1):
            self.one_loop(it)
        return self.logs

    def max_acq_function(self, index):
        tau = 1e-3
        configs = torch.Tensor(self.all_configs)
        preds = torch.stack([module(configs)[index].squeeze() for module in self.ensemble.modules])
        impr = preds - self.best_so_far[index].to(preds)
        qPI = torch.sigmoid(impr / tau).mean(dim=0)
        n_pts = self.n_full_evals_per_update if (index == 0) else self.n_part_evals_per_update
        mask = self.full_evals_mask if (index == 0) else self.part_evals_mask
        for idx in torch.argsort(qPI, descending=True):
            if mask[idx] == 0:
                mask[idx] = 1
                n_pts -= 1
            if n_pts == 0:
                break

    def evaluate_current_performance(self):
        self.current_cost = self.cost_fn(int(self.full_evals_mask.sum()), int(self.part_evals_mask.sum()))
        if self.full_evals_mask.sum():
            b_full = self.full_evals[self.full_evals_mask == 1].max()
            self.best_so_far[0] = b_full
            b_idx = self.full_evals[self.full_evals_mask == 1].argmax()
            self.best_so_far_cfg = self.all_configs[self.full_evals_mask == 1][b_idx].numpy()
        b_part = self.part_evals[self.part_evals_mask == 1].max()
        self.best_so_far[1] = b_part
        
        return {'cost': self.current_cost,
                'best_val': self.best_so_far[0],
                'best_solution': self.best_so_far_cfg}