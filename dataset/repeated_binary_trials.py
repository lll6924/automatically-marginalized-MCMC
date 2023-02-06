from numpyro.examples.datasets import BASEBALL, load_dataset
import jax.numpy as jnp

def get_repeated_binary_trials_data(dataset):
    if dataset == 'baseball_small':
        _, fetch_train = load_dataset(BASEBALL, split='train', shuffle=False)
        train, player_names = fetch_train()
        _, fetch_test = load_dataset(BASEBALL, split='test', shuffle=False)
        test, _ = fetch_test()
        return train[:, 0], train[:, 1], test[:, 0], test[:, 1]
    else:
        with open('data/'+dataset+'.txt') as f:
            lines = f.readlines()
            t1 = []
            t2 = []
            for line in lines:
                l = line.strip().split()
                t1.append(int(l[1]))
                t2.append(int(l[0]))
        return jnp.array(t1), jnp.array(t2), None, None

if __name__ == '__main__':
    print(get_repeated_binary_trials_data('baseball_large'))