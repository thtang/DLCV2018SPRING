import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_npy', type=str, default='save_full/para_dict.npy', help='dictionary filenameto be sparsifited')
    parser.add_argument('-o','--output_npy', type=str, default='save_finetune/sparse_dict.npy', help='sparsified dictionary filename')
    parser.add_argument('-th','--threshold', type=float, default=5e-2, help='cut-off threshold for small scaling factors')

    FLAG = parser.parse_args()
    
    print("===== sparsify network by scaling factors (gamma) =====")

    para_dict = np.load(FLAG.input_npy, encoding='latin1').item()
    sparse_dict, _ = myGammaSparsify(para_dict, thresh = FLAG.threshold)
    np.save(FLAG.output_npy, sparse_dict)
    print("===== save into {0} =====".format(FLAG.output_npy))

def myGammaSparsify(para_dict, thresh=0.05):
    last = None
    sparse_dict = {}
    N_total, N_remain = 0., 0.
    for k, v in sorted(para_dict.items()):
        if 'gamma' in k:
            # trim networks based on gamma
            gamma = v                      
            this = np.where(np.abs(gamma) > thresh)[0]
            sparse_dict[k] = gamma[this] 
            
            # get the layer name
            key = str.split(k,'_gamma')[0]
            
            # trim conv
            conv_, bias_ = para_dict[key]
            conv_ = conv_[:,:,:,this]
            if last is not None:
                conv_ = conv_[:,:,last,:]
            bias_ = bias_[this]
            sparse_dict[key] = [conv_, bias_]
            
            # get corresponding beta, bn_mean, bn_variance
            sparse_dict[key+"_beta"] = para_dict[key+"_beta"][this]
            sparse_dict[key+"_bn_mean"] = para_dict[key+"_bn_mean"][this]
            sparse_dict[key+"_bn_variance"] = para_dict[key+"_bn_variance"][this]
            
            # update
            last = this
            print('%s from %s to %s : %s ' % (k, len(gamma), len(this), len(this)/len(gamma)))
            N_total += len(gamma)
            N_remain += len(this)
    print('remain %s percentage' % (N_remain/N_total))
    W_, b_ = para_dict['fc_2']
    W_ = W_[last,:]
    sparse_dict['fc_2'] = [W_, b_]
    return sparse_dict, N_remain/N_total

if __name__ == '__main__':
    main()