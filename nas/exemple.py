import torch
from src.search_space import create_search_space, SearchInstance



# juste pour l'exemple
from itertools import product
from random import randint


if __name__ == '__main__':
    var_1_range = range(10)
    var_2_range = range(20)
    var_3_range = range(10)
    encodings = product(var_1_range, var_2_range, var_3_range)
    encodings = torch.Tensor(list(encodings))

    # Sans pré-entrainement
    search_space = create_search_space(name='Exemple',
                                       save_filename='test_search_space.dill',
                                       encodings=encodings,
                                       encoding_to_net=None,
                                       device='cpu')

    search_space.preprocess_no_pretraining()

    # Fonctions d'éval. Juste pour le test je vais mettre
    # la haute fidélité la somme des 3 variables
    # et la basse fidélité la somme + une petite perturbation aléatoire
    # ça prend une liste de encodings en entrée
    # Les costs sont importants juste pour l'affichage (la rapidité de la recherche)
    # en NAS je mets le nombre d'epochs dans l'éval à haute et à basse fidélité
    
    hi_fi_eval = lambda encodings_lst: [sum(encoding) for encoding in encodings_lst]
    hi_fi_cost = 200
    
    lo_fi_eval = lambda encodings_lst: [sum(encoding) + randint(-2,3) for encoding in encodings_lst]
    lo_fi_cost = 12

    search_instance = SearchInstance(name='Exemple',
                                     save_filename='test_search_inst.dill',
                                     search_space_filename='test_search_space.dill',
                                     hi_fi_eval = hi_fi_eval,
                                     hi_fi_cost = hi_fi_cost,
                                     lo_fi_eval = lo_fi_eval,
                                     lo_fi_cost = lo_fi_cost,
                                     device='cpu')

    search_instance.run_search(eval_budget=int(1e6))

    # ça va sauvegarder à chaque itération. si tu veux reprendre une recherche arrêtée,
    # load l'objet SearchInstance directement (avec dill), et lance avec run_search.
    # (pas besoin de redéfinir le search space etc..., il va être loadé du fichier)
    # with open('test_search_inst.dill', 'rb') as f:
    #    s = dill.load(f)