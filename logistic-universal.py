import argparse
import numpy as np
import time
from os import path
from warnings import simplefilter
import warnings
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
import random
from MLModel.optimizer import *
from MLModel.LoadData import *
from MLModel.MLmodel.logisticRegression import *
from MLModel.paramRange import *
from MLModel.LoadCoreset import *
from MLModel.hidden import *


def test(method='sgd', data='movieLen1M', exp_decay=1, subset_size=1., greedy=1, shuffle=0, g_cnt=-1.,
         b_cnt=-1., num_runs=10, metric='', reg=1e-5, rand='', ne=-1, from_all=0,
         coreset_from='scratch',
         batch=0, sampleSize=0):
    train_data, train_target, val_data, val_target, test_data, test_target = load_dataset(data)
    print("Dataset Loaded")
    print(np.unique(train_target))
    if data in ['IMDBC5', 'IMDBLargeC5', 'Brazilnew']:
        num_class = 5
    print("num class is ", num_class)
    print(np.unique(test_target))
    print("Class number is [{}]".format(num_class))
    g_range, b_range = get_param_range(subset_size, exp_decay, method, data)


    x_runs_a = [[]] * num_runs
    if ne == -1:
        ne = 20 + int(np.ceil((1. / subset_size) * 5)) + 5 if subset_size < 1 else 20
    else:
        rand += f'_e{ne}'
    if ne > 100:
        ne = 100
    f_runs_a = np.zeros((num_runs, ne))
    ft_runs_a = np.zeros((num_runs, ne))
    acc_runs_a = np.zeros((num_runs, ne))
    t_runs_a = np.zeros((num_runs, ne))


    precision_runs_a = np.zeros((num_runs, ne))
    recall_runs_a = np.zeros((num_runs, ne))

    best_f1_list = []
    best_f_list = []
    best_acc_list = []
    best_precision_list = []
    best_recall_list = []
    train_time_list = []
    best_MAE_list = []
    best_MSE_list = []
    best_MSLE_list = []

    for itr in range(num_runs):
        f_best, acc_best, b_f, g_f, b_a, g_a = 1e10, 0, 0, 0, 0, 0


        print("Cur itr is ", itr)
        # assert greedy == 1
        if greedy == 1:
            order, weights, total_ordering_time = LoadCoreset(coreset_from, data, subset_size, batch=batch, sampleSize=sampleSize)
        else:
            print('Selecting a random subset')

            order = np.arange(0, len(train_data))
            random.shuffle(order)
            order = order[:int(subset_size * len(train_data))]
            print(' 【Random subset size】 is ', int(subset_size * len(train_data)))
            weights = np.ones(int(subset_size * len(train_data)), dtype=np.float)

        print(f'--------------- run number: {itr}, rand: {rand}, '
              f'subset: {subset_size}, subset size: {len(order)}')

        best_test_f1 = 0
        best_test_acc = 0
        best_test_recall = 0
        best_test_precision = 0

        best_test_MAE = 0
        best_test_MSE = 0
        best_test_MSLE = 0

        print("g_range is ", g_range)
        print("b_range is ", b_range)
        for gamma in g_range:
            for b in b_range:


                dim = len(train_data[0])

                model = LogisticRegression(dim, num_class)
                lr = gamma * np.power(b, np.arange(ne)) if exp_decay else gamma / (1 + b * np.arange(ne))

                st_time = time.time()
                x_s, t_s = Optimizer().optimize(
                    method, model, train_data[order, :], train_target[order], weights, ne, shuffle, lr, reg)
                en_time = time.time()
                print("Train time is ", en_time - st_time)
                train_time_list.append(en_time - st_time)

                f_s = model.loss(train_data, train_target, l2_reg=reg)
                acc_s = model.accuracy(val_data, val_target)
                print('acc_s is ',acc_s)
                f1_s = model.f1(val_data, val_target)

                print(f'data: {data}, method: {method}, run: {itr}, exp_decay: {exp_decay}, size: {subset_size} {rand} '
                      f'--> f: {f_s}, acc: {acc_s}, f1: {f1_s} b: {b}, g: {gamma}')
                # if f1_s > f1_best:
                if acc_s > acc_best:

                    acc_best, x_a, g_a, b_a, t_a = acc_s, x_s, gamma, b, t_s

                    f1_best = f1_s
                    f_best = f_s

                    x_runs_a[itr] = x_a
                    t_runs_a[itr, :] = t_a
                    best_test_f1 = model.f1(test_data, test_target)
                    best_test_precision = model.precision(test_data, test_target)
                    best_test_recall = model.recall(test_data, test_target)
                    best_test_acc = model.acc(test_data, test_target)

                    best_test_MAE, best_test_MSE, best_test_MSLE = model.MASLE(test_data, test_target)
                    print("### New best MAE is ", best_test_MAE)
                    print("### New best MSE is ", best_test_MSE)
                    print("### New best MSLE is ", best_test_MSLE)

                    print("### New best f1 is [{}]".format(best_test_f1))
                    print("### New best precision is [{}]".format(best_test_precision))
                    print("### New best recall is [{}]".format(best_test_recall))
                    print("### New best acc is [{}]".format(best_test_acc))


            print(f'Best solution is => f: {f_best}, a: {acc_best}, b_f: {b_f}, g_f: {g_f}, b_a: {b_a}, g_a: {g_a}')


        best_f1_list.append(best_test_f1)
        best_f_list.append(f_best)
        best_acc_list.append(best_test_acc)
        best_precision_list.append(best_test_precision)
        best_recall_list.append(best_test_recall)

        best_MAE_list.append(best_test_MAE)
        best_MSE_list.append(best_test_MSE)
        best_MSLE_list.append(best_test_MSLE)

        print("   Current best_f1_list")
        print(best_f1_list)
        print("Mean ", np.mean(best_f1_list), "Max ", np.max(best_f1_list), "Min ", np.min(best_f1_list),
              "Median ", np.median(best_f1_list))

        print("   Current best f_list")
        print(best_f_list)
        print("Mean ", np.mean(best_f_list), "Max ", np.max(best_f_list), "Min ", np.min(best_f_list),
              "Median ", np.median(best_f_list))
        print("   Current best acc_list")
        print(best_acc_list)
        print("   Current best recall_list")
        print(best_recall_list)
        print("   Current best precision_list")
        print(best_precision_list)

        print("   Current best MAE_list")
        print(best_MAE_list)
        print("Mean ", np.mean(best_MAE_list), "Max ", np.max(best_MAE_list), "Min ", np.min(best_MAE_list),
              "Median ", np.median(best_MAE_list))

        print("   Current best MSE_list")
        print(best_MSE_list)
        print("Mean ", np.mean(best_MSE_list), "Max ", np.max(best_MSE_list), "Min ", np.min(best_MSE_list),
              "Median ", np.median(best_MSE_list))

        print("   Current best MSLE_list")
        print(best_MSLE_list)
        print("Mean ", np.mean(best_MSLE_list), "Max ", np.max(best_MSLE_list), "Min ", np.min(best_MSLE_list),
              "Median ", np.median(best_MSLE_list))

        print("Train time list(one hyper-param)")
        print(train_time_list)
        print("Mean ", np.mean(train_time_list), "Max ", np.max(train_time_list), "Min ", np.min(train_time_list), "Median ", np.median(train_time_list))
    print('Finish')

    return best_acc_list,best_MSE_list,train_time_list,best_f_list


if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Faster Training.')
    p.add_argument('--exp_decay', type=int, required=False, default=1,
                   choices=[0, 1], help='exponentially decaying learning rate')
    p.add_argument('--greedy', type=int, required=False, default=1,
                   help='greedy ordering')
    p.add_argument('--reg', type=float, required=False, default=1e-5,
                   help='L2 regularization constant')
    p.add_argument('--method', type=str, required=False, default='sgd',
                   choices=['sgd', 'svrg', 'saga', 'BGD'], help='sgd, svrg, saga, BGD')
    p.add_argument('--subset_size', '-s', type=float, required=False,
                   help='size of the subset')
    p.add_argument('--shuffle', type=int, default=2,
                   choices=[0, 1, 2, 3],
                   help='0: not shuffling, 1: random permutation, 2: with replacement, 3: fixed permutation')
    p.add_argument('--num_runs', type=int, required=False, default=10,
                   help='number of runs')
    p.add_argument('--data', type=str, required=False, default='IMDB',
                   choices=['IMDBC5','IMDBLargeC5', 'Brazilnew'], help='name of dataset')
    p.add_argument('--metric', type=str, required=False, default='l2',
                   help='distance metric')
    p.add_argument('--b', type=float, required=False, default=-1,
                   help='learning rate parameter b')
    p.add_argument('--g', type=float, required=False, default=-1,
                   help='learning rate parameter g')
    p.add_argument('--grad_diff', type=int, required=False, default=0,
                   help='number of epochs')
    p.add_argument('--from_all', type=int, required=False, default=0)
    p.add_argument('--coreset_from', type=str, required=False, default='diskOurs',
                   choices=['diskOurs'], help='Where to load coreset')
    p.add_argument('--batch', type=int, required=False, default=0)

    args = p.parse_args()

    if args.greedy == 0:
        rand = 'rand_nw'
    elif args.greedy == 1 and args.shuffle == 1:
        rand = 'grd_shuff'
    elif args.greedy == 1 and args.shuffle == 2:
        rand = 'grd_rand'
    elif args.greedy == 1 and args.shuffle == 0:
        rand = 'grd_ord'
    elif args.greedy == 1 and args.shuffle > 2:
        rand = 'grd_fix_perm'
    else:
        rand = ''

    print("Start test time", time.asctime( time.localtime(time.time()) ))
    test(method=args.method, data=args.data, exp_decay=args.exp_decay, subset_size=args.subset_size,
         greedy=args.greedy, shuffle=args.shuffle, b_cnt=args.b, g_cnt=args.g, num_runs=args.num_runs,
         metric=args.metric, rand=rand, ne=-1, from_all=args.from_all,
         coreset_from=args.coreset_from, reg=args.reg,batch=0)
    print("Finished test time", time.asctime(time.localtime(time.time()) ))



