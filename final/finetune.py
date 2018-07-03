import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import pickle
import imgaug as ia
from imgaug import augmenters as iaa

from VGG16_GAP import VGG16_GAP
from utils import readImgList, transformLabel, one_hot_encoding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--init_from', type=str, default='save_finetune/sparse_dict.npy', help='pre-trained weights')
    parser.add_argument('-c','--centers', type=str, default=None, help='parameters of centers')
    parser.add_argument('-s','--save_dir', type=str, default='save_finetune/', help='directory to store checkpointed models')
    parser.add_argument('-tp','--train_path', type=str, default='/home/cmchang/DLCV2018SPRING/final/dlcv_final_2_dataset/train/', help='path of training images')
    parser.add_argument('-tf','--train_file', type=str, default='/home/cmchang/DLCV2018SPRING/final/dlcv_final_2_dataset/train_id.txt', help='path of training id text files')
    parser.add_argument('-vp','--valid_path', type=str, default='/home/cmchang/DLCV2018SPRING/final/dlcv_final_2_dataset/val/', help='path of val images')
    parser.add_argument('-vf','--valid_file', type=str, default='/home/cmchang/DLCV2018SPRING/final/dlcv_final_2_dataset/val_id.txt', help='path of val id text files')
    parser.add_argument('-p' ,'--prof_type', type=str, default='all-one', help='type of profile coefficient')
    parser.add_argument('-ls','--lambda_s', type=float, default=0.0, help='coefficient of sparsity penalty')
    parser.add_argument('-lm','--lambda_m', type=float, default=0.0, help='coefficient of monotonicity-induced penalty')
    parser.add_argument('-lc','--lambda_c', type=float, default=1e-3, help='coefficient of center loss')
    parser.add_argument('-wd','--decay', type=float, default=1e-5, help='coefficient of weight decay')
    parser.add_argument('-lr','--learning_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('-kp','--keep_prob', type=float, default=1.0, help='dropout keep probability for fc layer')    

    FLAG = parser.parse_args()

    print("===== create directory =====")
    if not os.path.exists(FLAG.save_dir):
        os.makedirs(FLAG.save_dir)
    
    print("===== figiht =====")
    finetune(FLAG)

def finetune(FLAG):
    # already exist (uploaded to github)
    with open('save/label_dict.pkl', 'rb') as f:
        y_dict = pickle.load(f)
    
    dtrain = pd.read_csv(FLAG.train_file, header=None,sep=" ", names=["img", "id"])
    dvalid = pd.read_csv(FLAG.valid_file, header=None,sep=" ", names=["img", "id"])
    train_list = [os.path.join(FLAG.train_path, img) for img in list(dtrain.img)]
    valid_list = [os.path.join(FLAG.valid_path, img) for img in list(dvalid.img)]

    print("Reading train and valid images")
    Xtrain = readImgList(train_list)
    print("train: {0}".format(Xtrain.shape))
    Xvalid = readImgList(valid_list)
    print("valid: {0}".format(Xvalid.shape))

    ytrain = transformLabel(list(dtrain.id), y_dict)
    yvalid = transformLabel(list(dvalid.id), y_dict)

    Ytrain = one_hot_encoding(ytrain, len(y_dict))
    Yvalid = one_hot_encoding(yvalid, len(y_dict))


    print("Building model")
    scope_name = "Model"
    model = VGG16_GAP(scope_name=scope_name)

    model.build(vgg16_npy_path=FLAG.init_from,
                shape=Xtrain.shape[1:],
                classes=len(y_dict),
                prof_type=FLAG.prof_type,
                conv_pre_training=True,
                fc_pre_training=True,
                new_bn=False)

    if FLAG.centers is not None:
        centers = np.load(FLAG.centers)
        model.add_centers(centers.astype(np.float32))
    else:
        print("please specify your center.npy or initialize centers with all zeros")
        model.add_centers()
    

    print("Setting operations at various levels")
    dp = [1.0, 0.75, 0.5]
    tasks = [str(int(p*100)) for p in dp]
    model.set_idp_operation(dp=dp, decay=FLAG.decay, keep_prob=FLAG.keep_prob, lambda_c = FLAG.lambda_c)

    obj = 0.0
    for cur_task in tasks:
        print(cur_task)
        obj += model.loss_dict[cur_task]

    tracking = list()
    for cur_task in tasks:
        tracking.append(model.accu_dict[cur_task])

    # data augmenter
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    transform = iaa.Sequential([
        sometimes(iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)})),
        sometimes(iaa.Affine(scale={"x": (0.85, 1.15), "y":(0.85, 1.15)})),
        sometimes(iaa.Affine(rotate=(-45, 45))),
        sometimes(iaa.Fliplr(0.5))
    ])


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        augment = True

        # hyper parameters
        batch_size = 32
        epoch = 100
        early_stop_patience = 10
        min_delta = 0.0001

        # recorder
        epoch_counter = 0
        history = list()

        # Passing global_step to minimize() will increment it at each step.
        learning_rate = FLAG.learning_rate
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)

        checkpoint_path = os.path.join(FLAG.save_dir, 'model.ckpt')
        
        # trainable variables
        train_vars = list()
        for var in tf.trainable_variables():
            if model.scope_name in var.name:
                train_vars.append(var)

        for rm in model.gamma_var:
            train_vars.remove(rm)
            print('%s is not trainable.'% rm)
        
        for var in train_vars:
            if '_mean' in var.name:
                train_vars.remove(var)
                print('%s is not trainable.'% var)
        
        for var in train_vars:
            if '_beta' in var.name:
                train_vars.remove(var)
                print('%s is not trainable.'% var)
        
        for var in train_vars:
            if '_variance' in var.name:
                train_vars.remove(var)
                print('%s is not trainable.'% var)
        
        print(train_vars)
                
        train_op = opt.minimize(obj, var_list=train_vars)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=len(tasks))

        # max step in a epoch
        ptrain_max = int(Xtrain.shape[0]/batch_size)
        pval_max = int(Xvalid.shape[0]/batch_size)

        # re-initialize
        def initialize_uninitialized(sess):
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars): 
                    sess.run(tf.variables_initializer(not_initialized_vars))
        initialize_uninitialized(sess)

        # reset due to adding a new task
        patience_counter = 0
        current_best_val_accu = 0

        # optimize when the aggregated obj
        while(patience_counter < early_stop_patience and epoch_counter < epoch):
            
            # start training
            stime = time.time()
            train_loss, train_accu = 0.0, 0.0
            
            if augment:
                def load_batches():
                    for i in range(int(Xtrain.shape[0]/batch_size)):
                        print("Training: {0}/{1}".format(i,ptrain_max), end='\r')
                        st = i*batch_size
                        ed = (i+1)*batch_size
                        batch = ia.Batch(images=Xtrain[st:ed,:,:,:], data=Ytrain[st:ed,:])
                        yield batch

                batch_loader = ia.BatchLoader(load_batches)
                bg_augmenter = ia.BackgroundAugmenter(batch_loader=batch_loader, augseq=transform, nb_workers=1)

                while True:
                    batch = bg_augmenter.get_batch()
                    if batch is None:
                        print("Finished epoch.")
                        break
                    x_images_aug = batch.images_aug
                    y_images = batch.data
                    loss, accu, _, _ = sess.run([obj, model.accu_dict[cur_task], train_op, model.centers_update_op], 
                                            feed_dict={model.x: x_images_aug,
                                                        model.y: y_images,
                                                        model.is_train: True,
                                                        model.bn_train: False})
                    train_loss += loss
                    train_accu += accu
                batch_loader.terminate()
                bg_augmenter.terminate()
            else:
                for i in range(int(Xtrain.shape[0]/batch_size)):
                    print("Training: {0}/{1}".format(i,ptrain_max), end='\r')
                    st = i*batch_size
                    ed = (i+1)*batch_size
                    loss, accu, _, _ = sess.run([obj, model.accu_dict[tasks[0]], train_op, model.centers_update_op],
                                                        feed_dict={model.x: Xtrain[st:ed,:],
                                                                model.y: Ytrain[st:ed,:],
                                                                model.is_train: True,
                                                                model.bn_train: False})
                    train_loss += loss
                    train_accu += accu

            train_loss = train_loss/ptrain_max
            train_accu = train_accu/ptrain_max


            # validation
            val_loss, val_accu1, val_accu2 = 0.0, 0.0, 0.0
            val_accu_dp = list()
            for i in range(int(Xvalid.shape[0]/batch_size)):
                print("Validating: {0}/{1}".format(i,pval_max), end='\r')
                st = i*batch_size
                ed = (i+1)*batch_size
                loss, accu1, accu2, accu_dp = sess.run([obj, model.accu_dict[tasks[0]], model.accu_dict[tasks[-1]], tracking],
                                                    feed_dict={model.x: Xvalid[st:ed,:],
                                                            model.y: Yvalid[st:ed,:],
                                                            model.is_train: False,
                                                            model.bn_train: False})
                val_loss += loss
                val_accu1 += accu1
                val_accu2 += accu2
                val_accu_dp.append(accu_dp)
                
            val_accu_dp = np.mean(val_accu_dp, axis=0).tolist()
            dp_str = ""
            for i in range(len(tasks)):
                dp_str += "{0}%:{1}, ".format(tasks[i], np.round(val_accu_dp[i],4))
            
            print(dp_str)
            val_loss = val_loss/pval_max
            val_accu1 = val_accu1/pval_max
            val_accu2 = val_accu2/pval_max
            val_accu = val_accu1 # used for early stopping
            
            # early stopping check
            if (val_accu - current_best_val_accu) > min_delta:
                current_best_val_accu = val_accu
                patience_counter = 0

                para_dict = sess.run(model.para_dict)
                np.save(os.path.join(FLAG.save_dir, "para_dict.npy"), para_dict)
                print("save in %s" % os.path.join(FLAG.save_dir, "para_dict.npy"))
            else:
                patience_counter += 1

            # shuffle Xtrain and Ytrain in the next epoch
            idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[idx,:,:,:], Ytrain[idx,:]

            # epoch end
            epoch_counter += 1

            print("Epoch %s (%s), %s sec >> train loss: %.4f, train accu: %.4f, val loss: %.4f, val accu at %s: %.4f, val accu at %s: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), train_loss, train_accu, val_loss, tasks[0], val_accu1, tasks[-1], val_accu2))
            history.append([train_loss, train_accu, val_loss, val_accu])
            
            if epoch_counter % 10 == 0:
                import matplotlib.pyplot as plt
                df = pd.DataFrame(history)
                df.columns = ['train_loss', 'train_accu', 'val_loss', 'val_accu']
                df[['train_loss', 'val_loss']].plot()
                plt.savefig(os.path.join(FLAG.save_dir, 'loss.png'))
                plt.close()
                df[['train_accu', 'val_accu']].plot()
                plt.savefig(os.path.join(FLAG.save_dir, 'accu.png'))
                plt.close()
                
        saver.save(sess, checkpoint_path, global_step=epoch_counter)
        
        # extract features and calculate center

        output = []
        for i in range(int(Xtrain.shape[0]/200+1)):
            print(i, end="\r")
            st = i*200
            ed = min((i+1)*200, Xtrain.shape[0])
            prob = sess.run(model.features, feed_dict={model.x: Xtrain[st:ed,:], 
                                                        model.is_train: False,
                                                        model.bn_train: False})
            output.append(prob)

        for i in range(int(Xvalid.shape[0]/200+1)):
            print(i, end="\r")
            st = i*200
            ed = min((i+1)*200, Xvalid.shape[0])
            prob = sess.run(model.features, feed_dict={model.x: Xvalid[st:ed,:], 
                                                        model.is_train: False,
                                                        model.bn_train: False})
            output.append(prob)

        EX = np.concatenate(output)
        print(EX.shape)
        EY = np.concatenate([ytrain, yvalid])
        print(EY.shape)
        centers = np.zeros((len(y_dict), EX.shape[1]))
        for i in range(len(y_dict)):
            centers[i,:] = np.mean(EX[EY==i,:], axis=0)
            np.save(arr=centers,file=os.path.join(FLAG.save_dir,"centers.npy"))

if __name__ == '__main__':
    main()