import tensorflow as tf
import numpy as np

losses = []

def get_loss(loss):
    losses.append(loss)
	
def add_layer(L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX):
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX],stddev=stddev),name = 'w')
		with tf.name_scope('biases'):
			biases_LX = tf.Variable(tf.zeros([1,num_nodes_LX]),name = 'b')
		tf.add_to_collection("p_var",Weights_LX)
		tf.add_to_collection("p_var",biases_LX)
		with tf.name_scope('xW_plus_b'):
			xW_plus_b_LX = tf.matmul(L_Prev,Weights_LX)+biases_LX
		if activation_LX is None:
			LX = xW_plus_b_LX
		else:
			LX = activation_LX(xW_plus_b_LX)

		return LX,Weights_LX,biases_LX 
    


#lambd = 0.5
stddev=0.05

TotTrainData =1970 
num_iters = 15000


dim_input1 = 32*32
dim_sen=32*32
dim_output = 1
Nlayer=5

dataset = np.load('dens_train.npy')
dataset2 = np.load('sensi1_train.npy')
dataset3 = np.load('obj_train.npy')

dataset=dataset[0:(TotTrainData)][:]
dataset2=dataset2[0:(TotTrainData)][:]
dataset3=dataset3[0:(TotTrainData),0].reshape((TotTrainData),dim_output)


max3=1.1e9
min3=0.0
dataset2=2.0*dataset2/(max3-min3)
dataset3=2.0*(dataset3-min3)/(max3-min3)-1.0


x_data = dataset[0:TotTrainData][:]
y2_data = dataset2[0:TotTrainData][:]
y_data = dataset3[0:TotTrainData,0].reshape(TotTrainData,dim_output)


######################################################################


with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32,[None,dim_input1],name='x_input')
	y2 = tf.placeholder(tf.float32,[None,dim_sen],name='y21_input')
	y = tf.placeholder(tf.float32,[None,dim_output],name='y_input')




L1,Weights_L1,biases_L1 = add_layer(x,dim_input1,32*32,tf.nn.tanh)
L2,Weights_L2,biases_L2 = add_layer(L1,32*32,8*32,tf.nn.tanh)
L3,Weights_L3,biases_L3 = add_layer(L2,8*32,32,tf.nn.tanh)


Weights_L4 = tf.Variable(1*tf.random_normal([32,dim_output],stddev=stddev))
biases_L4 = tf.Variable(tf.zeros([1,dim_output]))


tf.add_to_collection("p_var",Weights_L4)
tf.add_to_collection("p_var",biases_L4)

L4 = tf.matmul(L3,Weights_L4)+biases_L4
    

prediction1 =L4
prediction2=tf.gradients(L4,x)


with tf.name_scope('loss'):
	loss = 1*(tf.reduce_sum(tf.square(y-prediction1)))+1e6*(tf.reduce_sum(tf.square(y2-prediction2)))


def SecondOrderOptimizer():
    with tf.name_scope('train'):
        train = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': num_iters})
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection_ref("p_var"))
        if False:
           saver.restore(sess,'./savemodel1/model')
        train.minimize(sess, fetches=[loss],loss_callback=get_loss,feed_dict={x:x_data,y2:y2_data,y:y_data})
        saver.save(sess, "./savemodel1/model")


SecondOrderOptimizer()
