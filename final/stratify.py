import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_npy', type=str, default='para_dict.npy', help='dictionary filenameto be sparsifited')
    parser.add_argument('-o','--output_npy', type=str, default='sparse_dict.npy', help='sparsified dictionary filename')
    parser.add_argument('-p','--percentage', type=float, default=5e-2, help='cut-off threshold for small scaling factors')

    FLAG = parser.parse_args()
    
    print("===== stratify network by a given percentage =====")

    para_dict = np.load(FLAG.input_npy, encoding='latin1').item()
    stratified_dict = myPercentStratify(para_dict, dp = FLAG.percentage)
    np.save(FLAG.output_npy, stratified_dict)
    print("===== save into {0} =====".format(FLAG.output_npy))

def myPercentStratify(para_dict, dp):
    """
    dp: usage percentage of channels in each layer
    """
    first = True
    new_dict = {}
    last = 3
    for k,v in sorted(para_dict.items()):
        if 'bn_mean' in k:
            new_dict[k] = v[:int(len(v)*dp)]
            print("%s:%s" % (k, new_dict[k].shape))
        elif 'bn_variance' in k:
            new_dict[k] = v[:int(len(v)*dp)]
            print("%s:%s" % (k, new_dict[k].shape))
        elif 'gamma' in k:
            new_dict[k] = v[:int(len(v)*dp)]
            print("%s:%s" % (k, new_dict[k].shape))
        elif 'beta' in k:
            new_dict[k] = v[:int(len(v)*dp)]
            print("%s:%s" % (k, new_dict[k].shape))
        elif 'conv' in k:
            O = v[0].shape[3]
            new_dict[k] = v[0][:,:,:last,:int(O*dp)], v[1][:int(O*dp)]
            last = int(O*dp)
            print("W%s:%s" % (k, new_dict[k][0].shape))
            print("b%s:%s" % (k, new_dict[k][1].shape))
        elif 'fc_2' in k:
            O = v[0].shape[0]
            new_dict[k] = [v[0][:int(O*dp),:], v[1]]
        else:
            new_dict[k] = v
    return new_dict

if __name__ == '__main__':
    main()