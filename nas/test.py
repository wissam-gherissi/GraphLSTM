import dill

if __name__== '__main__':
    with open('test_search_inst.dill', 'rb') as f:
        logs = dill.load(f).logs
    best_solution = logs[-1]['best_solution']
    print(best_solution)
