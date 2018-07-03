import os
import time
import numpy as np
import tensorflow as tf

class VGG16_GAP:
    def __init__(self, scope_name="VGG16"):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """
        
        self.scope_name = scope_name
        
        self.gamma_var = []
        self.net_shape = []
        
        # operation dictionary
        self.prob_dict = {}
        self.loss_dict = {}
        self.accu_dict = {}

        # parameter dictionary
        self.para_dict = {}

    def build(self,
              vgg16_npy_path=None, 
              classes=10, 
              shape=(32,32,3), 
              prof_type=None, 
              conv_pre_training=True, 
              fc_pre_training=True,
              new_bn=True):
        """
        This function defines all the variables in a model for building computation graphs later.
        
        @param vgg16_npy_path    : either a path of model npy or a dictionary of weights
        @param classes           : number of output class
        @param shape             : input shape
        @param prof_type         : monotonically decreasing basic function, {'linear', 'all-one','half-exp','harmonic'}
        @param conv_pre_training : whether use pre-trained weights of conv_layers in vgg16_npy_path
        @param fc_pre_training   : whether use pre-trained weights of fc_layers in vgg16_npy_path
        @param new_bn            : whether renew the batch normalization parameters of conv_layer in vgg16_npy_path
        """
        
        # input information
        self.H, self.W, self.C = shape
        self.classes = classes
        
        start_time = time.time()
        print("build model started")

        if prof_type is None:
            self.prof_type = "all-one"
        else:
            self.prof_type = prof_type

        # load pre-trained weights
        if isinstance(vgg16_npy_path,dict):
            self.data_dict = vgg16_npy_path
            print("parameters loaded")
        elif isinstance(vgg16_npy_path, str):
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = dict()
            print("no npy file")

        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.H, self.W, self.C])
        self.y = tf.placeholder(tf.float32, [None, self.classes])
        self.is_train = tf.placeholder(tf.bool)
        self.bn_train = tf.placeholder(tf.bool)
        
        self.x = self.x/255.0
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]

        # the value only used when not leveraging pre-trained weights
        vgg16_conv_layer = {'conv1_1':(3, 3, 3, 64),
                            'conv1_2':(3, 3, 64, 64),
                            'conv2_1':(3, 3, 64, 128),
                            'conv2_2':(3, 3, 128, 128),
                            'conv3_1':(3, 3, 128, 256),
                            'conv3_2':(3, 3, 256, 256),
                            'conv3_3':(3, 3, 256, 256),
                            'conv4_1':(3, 3, 256, 512),
                            'conv4_2':(3, 3, 512, 512),
                            'conv4_3':(3, 3, 512, 512),
                            'conv5_1':(3, 3, 512, 512),
                            'conv5_2':(3, 3, 512, 512),
                            'conv5_3':(3, 3, 512, 512)}
        my_fc_layer = {'fc_2':(512, self.classes)}

        # declare and initialize the weights of VGG16
        with tf.variable_scope(self.scope_name):
            # weight decay
            self._weight_decay = 0.0
            for k, v in vgg16_conv_layer.items():
                if conv_pre_training:
                    conv_filter, gamma, beta, bn_mean, bn_variance = self.get_conv_filter(name=k, new_bn=new_bn)
                    conv_bias =  self.get_bias(name=k)
                else:
                    conv_filter, gamma, beta, bn_mean, bn_variance = self.get_conv_filter(name=k, shape=v)
                    conv_bias = self.get_bias(name=k, shape=(v[3],))
                
                self.para_dict[k] = [conv_filter, conv_bias]
                self.para_dict[k+"_gamma"] = gamma
                self.para_dict[k+"_beta"] = beta
                self.para_dict[k+"_bn_mean"] = bn_mean
                self.para_dict[k+"_bn_variance"] = bn_variance
                self.gamma_var.append(self.para_dict[k+"_gamma"])
                
                # weight decay
                self._weight_decay += tf.nn.l2_loss(conv_filter)+tf.nn.l2_loss(conv_bias)
            
            for k, v in my_fc_layer.items():
                if fc_pre_training:
                    fc_W = self.get_fc_layer(name=k)
                    fc_b = self.get_bias(name=k)
                else:
                    fc_W = self.get_fc_layer(name=k, shape=v)
                    fc_b = self.get_bias(name=k, shape=(v[1],))
            
            self.para_dict['fc_2'] = [fc_W, fc_b]
            self._weight_decay += tf.nn.l2_loss(fc_W) + tf.nn.l2_loss(fc_b)
        
        print(("build model finished: %ds" % (time.time() - start_time)))
        
    def add_centers(self, centers=None):
        """
        add centers of every class
        
        @param centers: its shape must be (# of classes, # of features)
        """
        with tf.variable_scope(self.scope_name):
            # classes vs. feature
            if centers is not None:
                self.centers = tf.get_variable(initializer=centers, name="centers", dtype=tf.float32, trainable=False)
            else:
                self.centers = tf.get_variable(shape=(self.classes, self.para_dict['fc_2'][0].shape[0]), initializer=tf.zeros_initializer(), name='centers', dtype=tf.float32, trainable=False)

    def sparsity_train(self, l1_gamma=0.001, l1_gamma_diff=0.001, decay=0.0005, keep_prob=0.0, lambda_c = 1e-4):
        """
        define computational graphs for training a compact model.
        
        @param l1_gamma      : the coefficient of sparsity penalty (float)
        @param l1_gamma_diff : the coefficient of monotonicity-induced penalty (float)
        @param decay         : the coefficient of weight decay (float)
        @param keep_prob     : keep_prob of dropout layer (float)
        @param lambda_c      : the coefficient of lambda_c (float)
        """
        start_time = time.time()
        with tf.name_scope("var_dp"):
            conv1_1 = self.idp_conv_bn_layer( self.x, "conv1_1")
            conv1_2 = self.idp_conv_bn_layer(conv1_1, "conv1_2")
            pool1 = self.max_pool(conv1_2, 'pool1')

            conv2_1 = self.idp_conv_bn_layer(  pool1, "conv2_1")
            conv2_2 = self.idp_conv_bn_layer(conv2_1, "conv2_2")
            pool2 = self.max_pool(conv2_2, 'pool2')

            conv3_1 = self.idp_conv_bn_layer(  pool2, "conv3_1")
            conv3_2 = self.idp_conv_bn_layer(conv3_1, "conv3_2")
            conv3_3 = self.idp_conv_bn_layer(conv3_2, "conv3_3")
            pool3 = self.max_pool(conv3_3, 'pool3')

            conv4_1 = self.idp_conv_bn_layer(  pool3, "conv4_1")
            conv4_2 = self.idp_conv_bn_layer(conv4_1, "conv4_2")
            conv4_3 = self.idp_conv_bn_layer(conv4_2, "conv4_3")
            pool4   = self.max_pool(conv4_3, 'pool4')

            conv5_1 = self.idp_conv_bn_layer(  pool4, "conv5_1")
            conv5_2 = self.idp_conv_bn_layer(conv5_1, "conv5_2")
            conv5_3 = self.idp_conv_bn_layer(conv5_2, "conv5_3")
            pool5 = self.global_avg_pool(conv5_3, 'pool5')
            
            # features
            self.features = pool5
            
            # dropout
            pool5 = self.dropout_layer(pool5, keep_prob)
            
            # logit
            logits = self.fc_layer(pool5, 'fc_2')       
            prob = tf.nn.softmax(logits, name="prob")
            
            # cross_entropy loss
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
            loss = tf.reduce_mean(cross_entropy)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(x=tf.argmax(logits, 1), y=tf.argmax(self.y, 1)),tf.float32))
            
            # center loss
            labels = tf.argmax(self.y, 1)
            batch_centers = tf.gather(self.centers, labels, axis=0) # batch,
            self.center_loss = tf.nn.l2_loss(self.features - batch_centers) # tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(x=self.features, y=batch_centers)), axis=1))                
            
            # update centers using this batch samples
            diff = batch_centers - self.features
            unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff = diff / tf.cast((1 + appear_times), tf.float32)
            diff = 0.5 * diff
            self.centers_update_op = tf.scatter_sub(self.centers, labels, diff)
            
            # sparsity penalty: imposing l1 regularization upon gamma
            l1_gamma_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_gamma)
            gamma_l1 = tf.contrib.layers.apply_regularization(l1_gamma_regularizer, self.gamma_var)

            # monotonicity-induced penalty: imposing l1 regularization upon gamma_diff
            def non_increasing_constraint_axis_0(a):
                return tf.nn.relu(a[1:]-a[:-1])
            gamma_diff_var = [non_increasing_constraint_axis_0(x) for x in self.gamma_var]
            l1_gamma_diff_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_gamma_diff)
            gamma_diff_l1 = tf.contrib.layers.apply_regularization(l1_gamma_diff_regularizer, gamma_diff_var)
            
            # store sparsity_train related operations into dictionary
            self.prob_dict["var_dp"] = prob
            self.loss_dict["var_dp"] = loss + gamma_l1 + gamma_diff_l1 + self._weight_decay * decay + self.center_loss*lambda_c
            self.accu_dict["var_dp"] = accuracy
     
        print(("sparsity train operation setup: %ds" % (time.time() - start_time)))
    
    def set_idp_operation(self, dp, decay=0.0, keep_prob=1.0, lambda_c = 1e-4, train=True):
        """
        define computational graphs at different utilization levels.
        
        @param dp        : percentage of utilization (list)
        @param decay     : the coefficient of weight decay (float)
        @param keep_prob : keep_prob of dropout layer (float)
        @param lambda_c  : the coefficient of lambda_c (float)
        """
        if type(dp) != list:
            raise ValueError("dp must be a list containing utilization levels of interests.")
        else:
            self.dp = dp 
            print("DP under test:", np.round(self.dp,2))
            
        start_time = time.time()
        # create operations at every dot product percentages
        for dp_i in dp:
            with tf.name_scope(str(int(dp_i*100))):
                conv1_1 = self.idp_conv_bn_layer( self.x, "conv1_1", dp_i)
                conv1_2 = self.idp_conv_bn_layer(conv1_1, "conv1_2", dp_i)
                pool1 = self.max_pool(conv1_2, 'pool1')

                if dp_i == 1.0:
                    self.net_shape.append(conv1_1.get_shape())
                    self.net_shape.append(pool1.get_shape())

                conv2_1 = self.idp_conv_bn_layer(  pool1, "conv2_1", dp_i)
                conv2_2 = self.idp_conv_bn_layer(conv2_1, "conv2_2", dp_i)
                pool2 = self.max_pool(conv2_2, 'pool2')

                if dp_i == 1.0:
                    self.net_shape.append(conv2_1.get_shape())
                    self.net_shape.append(pool2.get_shape())

                conv3_1 = self.idp_conv_bn_layer(  pool2, "conv3_1", dp_i)
                conv3_2 = self.idp_conv_bn_layer(conv3_1, "conv3_2", dp_i)
                conv3_3 = self.idp_conv_bn_layer(conv3_2, "conv3_3", dp_i)
                pool3 = self.max_pool(conv3_3, 'pool3')

                if dp_i == 1.0:
                    self.net_shape.append(conv3_1.get_shape())
                    self.net_shape.append(conv3_2.get_shape())
                    self.net_shape.append(pool3.get_shape())

                conv4_1 = self.idp_conv_bn_layer(  pool3, "conv4_1", dp_i)
                conv4_2 = self.idp_conv_bn_layer(conv4_1, "conv4_2", dp_i)
                conv4_3 = self.idp_conv_bn_layer(conv4_2, "conv4_3", dp_i)
                pool4 = self.max_pool(conv4_3, 'pool4')
                
                if dp_i == 1.0:
                    self.net_shape.append(conv4_1.get_shape())
                    self.net_shape.append(conv4_2.get_shape())
                    self.net_shape.append(pool4.get_shape())

                conv5_1 = self.idp_conv_bn_layer(  pool4, "conv5_1", dp_i)
                conv5_2 = self.idp_conv_bn_layer(conv5_1, "conv5_2", dp_i)
                conv5_3 = self.idp_conv_bn_layer(conv5_2, "conv5_3", dp_i)
                pool5 = self.global_avg_pool(conv5_3, 'pool5')

                if dp_i == 1.0:
                    self.net_shape.append(conv5_1.get_shape())
                    self.net_shape.append(conv5_2.get_shape())
                    self.net_shape.append(pool5.get_shape())
                    # features
                    self.features = pool5
                
                pool5 = self.dropout_layer(pool5, keep_prob)
                
                logits = self.fc_layer(pool5, 'fc_2')
                prob = tf.nn.softmax(logits, name="prob")

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
                loss = tf.reduce_mean(cross_entropy)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(x=tf.argmax(logits, 1), y=tf.argmax(self.y, 1)), dtype=tf.float32))
                
                # center loss
                labels = tf.argmax(self.y, 1)
                batch_centers = tf.gather(self.centers, labels, axis=0) # batch,
                self.center_loss = tf.nn.l2_loss(self.features - batch_centers) # tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(x=self.features, y=batch_centers)), axis=1))                
                
                diff = batch_centers - self.features
                unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
                appear_times = tf.gather(unique_count, unique_idx)
                appear_times = tf.reshape(appear_times, [-1, 1])

                diff = diff / tf.cast((1 + appear_times), tf.float32)
                diff = 0.5 * diff
                self.centers_update_op = tf.scatter_sub(self.centers, labels, diff)

                # self.feature_dict[str(int(dp_i*100))] = fc_1
                self.prob_dict[str(int(dp_i*100))] = prob
                self.accu_dict[str(int(dp_i*100))] = accuracy
                self.loss_dict[str(int(dp_i*100))] = loss + self._weight_decay * decay + self.center_loss*lambda_c

        print(("Set dp operations finished: %ds" % (time.time() - start_time)))

    def spareness(self, thresh=0.05):
        N_active, N_total = 0,0
        for gamma in self.gamma_var:
            m = tf.cast(tf.less(tf.abs(gamma), thresh), tf.float32)
            n_active = tf.reduce_sum(m)
            n_total  = tf.cast(tf.reduce_prod(tf.shape(m)), tf.float32)
            N_active += n_active
            N_total  += n_total
        return N_active/N_total
    
    def global_avg_pool(self, bottom, name):
        return tf.reduce_mean(bottom, axis=[1,2], name=name)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def dropout_layer(self, bottom, keep_prob):
        if self.is_train == True:
            return tf.nn.dropout(bottom, keep_prob=keep_prob)
        else:
            return bottom

    def idp_conv_bn_layer(self, bottom, name, dp=1.0):
        with tf.name_scope(name+str(int(dp*100))):
            with tf.variable_scope(self.scope_name,reuse=True):
                conv_filter = tf.get_variable(name=name+"_W")
                conv_biases = tf.get_variable(name=name+"_b")
                conv_gamma  = tf.get_variable(name=name+"_gamma")
                moving_mean = tf.get_variable(name=name+'_bn_mean')
                moving_variance = tf.get_variable(name=name+'_bn_variance')
                beta = tf.get_variable(name=name+'_beta')
            H,W,C,O = conv_filter.get_shape().as_list()
            # print(bottom.get_shape().as_list())
            
            # ignore input images
            if name is not 'conv1_1':
                bottom = bottom[:,:,:,:int(C*dp)]
                # print("AFTER",bottom.get_shape().as_list())
                conv_filter = conv_filter[:,:,:int(C*dp),:]

            # create a mask determined by the dot product percentage
            n1 = int(O * dp)
            n0 = O - n1
            mask = tf.constant(value=np.append(np.ones(n1, dtype='float32'), np.zeros(n0, dtype='float32')), dtype=tf.float32)
            conv_gamma = tf.multiply(conv_gamma, mask)
            beta = tf.multiply(beta, mask)
            
            conv = tf.nn.conv2d(bottom, conv_filter, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, conv_biases)

            from tensorflow.python.training.moving_averages import assign_moving_average
            def mean_var_with_update():
                mean, variance = tf.nn.moments(conv, [0,1,2], name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, 0.99),
                                              assign_moving_average(moving_variance, variance, 0.99)]):
                    return tf.identity(mean), tf.identity(variance)

            mean, variance = tf.cond(self.bn_train, mean_var_with_update, lambda:(moving_mean, moving_variance))

            conv = tf.nn.batch_normalization(conv, mean, variance, beta, conv_gamma, 1e-05)
            relu = tf.nn.relu(conv)
            
            return relu

    def fc_layer(self, bottom, name):
        with tf.name_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            
            with tf.variable_scope(self.scope_name,reuse=True):
                weights = tf.get_variable(name=name+"_W")
                biases = tf.get_variable(name=name+"_b")

            # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_filter(self, name, new_bn=False, shape=None):
        if shape is not None:
            conv_filter = tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_W", dtype=tf.float32)
        elif name in self.data_dict.keys():
            conv_filter = tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W")
        else:
            print("please specify a name in data_dict or specify a shape in use")

        H,W,C,O = conv_filter.get_shape().as_list()

        if name+"_gamma" in self.data_dict.keys() and not new_bn: 
            gamma = tf.get_variable(initializer=self.data_dict[name+"_gamma"], name=name+"_gamma")
        else:
            gamma = tf.get_variable(initializer=self.get_profile(O, self.prof_type), name=name+"_gamma")

        if name+"_beta" in self.data_dict.keys() and not new_bn:
            beta = tf.get_variable(initializer=self.data_dict[name+"_beta"], name=name+"_beta")
        else:
            beta = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_beta')

        if name+"_bn_mean" in self.data_dict.keys() and not new_bn:
            bn_mean = tf.get_variable(initializer=self.data_dict[name+"_bn_mean"], name=name+"_bn_mean")
        else:
            bn_mean = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_bn_mean')

        if name+"_bn_variance" in self.data_dict.keys() and not new_bn: 
            bn_variance = tf.get_variable(initializer=self.data_dict[name+"_bn_variance"], name=name+"_bn_variance")
        else:
            bn_variance = tf.get_variable(shape=(O,),initializer=tf.ones_initializer(), name=name+'_bn_variance')
        
        return conv_filter, gamma, beta, bn_mean, bn_variance

    def get_fc_layer(self, name, shape=None):
        if shape is not None:
            return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_W", dtype=tf.float32)
        elif name in self.data_dict.keys():
            return tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W", dtype=tf.float32)
        else:
            print("please specify a name in data_dict or specify a shape in use")
            return None
            
    def get_bias(self, name, shape=None):
        if shape is not None:
            return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_b", dtype=tf.float32)
        elif name in self.data_dict.keys():
            return tf.get_variable(initializer=self.data_dict[name][1], name=name+"_b", dtype=tf.float32)
        else:
            print("please specify a name in data_dict or specify a shape in use")
            return None

    def get_profile(self, C, prof_type):
        def half_exp(n, k=1, dtype='float32'):
            n_ones = int(n/2)
            n_other = n - n_ones
            return np.append(np.ones(n_ones, dtype=dtype), np.exp((1-k)*np.arange(n_other), dtype=dtype))
        if prof_type == "linear":
            profile = np.linspace(2.0,0.0, num=C, endpoint=False, dtype='float32')
        elif prof_type == "all-one":
            profile = np.ones(C, dtype='float32')
        elif prof_type == "half-exp":
            profile = half_exp(C, 2.0)
        elif prof_type == "harmonic":
            profile = np.array(1.0/(np.arange(C)+1), dtype='float32')
        else:
            raise ValueError("prof_type must be \"all-one\", \"half-exp\", \"harmonic\" or \"linear\".")
        return profile
                
