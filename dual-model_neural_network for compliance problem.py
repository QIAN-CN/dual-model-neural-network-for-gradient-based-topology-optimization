import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import math
tf.reset_default_graph()
losses = []
from tensorflow import set_random_seed
num_iters=5000
batch_size=500
ave=0.00
sigma=0.05
c_mid=0.0018
scale=0.0012
d_num=1929 
test_num=100
valid_num=100
dpn=100
input_dim=36
output_dim=1
dens_train = (np.load('dens_train.npy')-0.5)
fy_train = (np.load('fy_train.npy')-0.5)
comp_train=(np.load('comp_train.npy')-c_mid)/scale
comp_grad_train=np.load('comp_grad_train.npy')/scale
dataset0=np.zeros((d_num,input_dim,input_dim,2))
dataset2=np.zeros((d_num,input_dim,input_dim,1))
for i in range(d_num):
    dataset0_flag=np.array(dens_train[i]).reshape(input_dim,input_dim)
    fy0_flag=np.array(fy_train[i]).reshape(input_dim,input_dim)
    dataset2_flag=np.array(comp_grad_train[i]).reshape(input_dim,input_dim)
    for j in range(input_dim):
        for k in range(input_dim):
            dataset0[i][j][k][0]=dataset0_flag[j][k]
            dataset0[i][j][k][1]=fy0_flag[j][k]
            dataset2[i][j][k][0]=dataset2_flag[j][k]
dataset1 = np.array(comp_train).reshape(comp_train.shape[0],1)


x_data=np.array(dataset0[0:d_num,:])
y_data=np.array(dataset1[0:d_num,:])
yg_data=np.array(dataset2[0:d_num,:])

x_data_pre=np.array(x_data[0:test_num,:]) #just for show
y_data_pre=np.array(y_data[0:test_num,:]) #just for show
yg_data_pre=np.array(yg_data[0:test_num,:]) #just for show

del dataset0
del dataset1
del dataset2
del dens_train
del fy_train
del comp_train
del comp_grad_train

dens_test = (np.load('dens_test.npy')-0.5)
fy_test = (np.load('fy_test.npy')-0.5)
comp_test=(np.load('comp_test.npy')-c_mid)/scale
comp_grad_test=np.load('comp_grad_test.npy')/scale

dataset3=np.zeros((test_num,input_dim,input_dim,2))
dataset5=np.zeros((test_num,input_dim,input_dim,1))
for i in range(test_num):
    dataset3_flag=np.array(dens_test[i]).reshape(input_dim,input_dim)
    fy3_flag=np.array(fy_test[i]).reshape(input_dim,input_dim)
    dataset5_flag=np.array(comp_grad_test[i]).reshape(input_dim,input_dim)
    for j in range(input_dim):
        for k in range(input_dim):
            dataset3[i][j][k][0]=dataset3_flag[j][k]
            dataset3[i][j][k][1]=fy3_flag[j][k]
            dataset5[i][j][k][0]=dataset5_flag[j][k]
dataset4 = np.array(comp_test).reshape(comp_test.shape[0],1)


x_test=np.array(dataset3[0:test_num,:])
y_test=np.array(dataset4[0:test_num,:])
yg_test=np.array(dataset5[0:test_num,:])

del dataset3
del dataset4
del dataset5
del dens_test
del fy_test
del comp_test
del comp_grad_test

dens_valid = (np.load('dens_valid.npy')-0.5)
fy_valid = (np.load('fy_valid.npy')-0.5)
comp_valid=(np.load('comp_valid.npy')-c_mid)/scale
comp_grad_valid=np.load('comp_grad_valid.npy')/scale
dataset6=np.zeros((test_num,input_dim,input_dim,2))
dataset8=np.zeros((test_num,input_dim,input_dim,1))
for i in range(test_num):
    dataset6_flag=np.array(dens_valid[i]).reshape(input_dim,input_dim)
    fy6_flag=np.array(fy_valid[i]).reshape(input_dim,input_dim)
    dataset8_flag=np.array(comp_grad_valid[i]).reshape(input_dim,input_dim)
    for j in range(input_dim):
        for k in range(input_dim):
            dataset6[i][j][k][0]=dataset6_flag[j][k]
            dataset6[i][j][k][1]=fy6_flag[j][k]            
            dataset8[i][j][k][0]=dataset8_flag[j][k]
dataset7 = np.array(comp_valid).reshape(comp_valid.shape[0],1)

x_valid=np.array(dataset6[0:test_num,:])
y_valid=np.array(dataset7[0:test_num,:])
yg_valid=np.array(dataset8[0:test_num,:])

del dataset6
del dataset7
del dataset8

del dens_valid
del comp_valid
del comp_grad_valid


x = tf.placeholder(tf.float32, [None, input_dim, input_dim, 2])  
y = tf.placeholder(tf.float32, [None,1])
yg= tf.placeholder(tf.float32, [None,input_dim, input_dim,1])

def my_conv2d(x,in_ch,out_ch,me,std,ker_sh1,ker_sh2,str1,str2,activation_L1,pad):
    kernel=tf.Variable(tf.random_normal(shape=[ker_sh1,ker_sh2,in_ch,out_ch],mean = me,stddev = std))
    b=tf.Variable(tf.zeros([out_ch]))
    tf.add_to_collection("p_var",kernel)
    tf.add_to_collection("p_var",b)
    if activation_L1 is None:
        L = tf.nn.conv2d(x,kernel,strides=[1,str1,str2,1],padding=pad)+b
    else:  
        L = activation_L1(tf.nn.conv2d(x,kernel,strides=[1,str1,str2,1],padding=pad)+b)
    return L


def my_conv2d_transpose(x,out_wid,out_hei,in_ch,out_ch,me,std,ker_sh1,ker_sh2,str1,str2,activation_L1):
    kernel=tf.Variable(tf.random_normal(shape=[ker_sh1,ker_sh2,in_ch,out_ch],mean = me,stddev = std))
    output_shape=[tf.shape(x)[0],out_wid,out_hei,in_ch]
    #print(output_shape)
    b=tf.Variable(tf.zeros([in_ch]))
    tf.add_to_collection("p_var",kernel)
    tf.add_to_collection("p_var",b)
    if activation_L1 is None:
        L = tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME")+b
    else:
        L =tf.add(activation_L1(tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME")+b),tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME"))
    return L



def add_layer(L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX):
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX],stddev=sigma),name = 'w')  
			tf.add_to_collection("p_var",Weights_LX)
		with tf.name_scope('biases'):
			biases_LX = tf.Variable(tf.zeros([1,num_nodes_LX]),name = 'b')
			tf.add_to_collection("p_var",biases_LX)  
		with tf.name_scope('xW_plus_b'):
			xW_plus_b_LX = tf.matmul(L_Prev,Weights_LX)+biases_LX
		if activation_LX is None:
			LX = xW_plus_b_LX
		else:
			LX = activation_LX(xW_plus_b_LX)
		return LX

def add_layer_with_res(L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX):
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX],stddev=sigma),name = 'w')  
			tf.add_to_collection("p_var",Weights_LX)
		with tf.name_scope('biases'):
			biases_LX = tf.Variable(tf.zeros([1,num_nodes_LX]),name = 'b')
			tf.add_to_collection("p_var",biases_LX)  
		with tf.name_scope('xW_plus_b'):
			xW_plus_b_LX = tf.matmul(L_Prev,Weights_LX)+biases_LX
		if activation_LX is None:
			LX = xW_plus_b_LX
		else:
			LX = tf.add(activation_LX(xW_plus_b_LX),xW_plus_b_LX)
		return LX
        

y1=my_conv2d(x,2,3,ave,sigma,5,5,1,1,tf.nn.tanh,"SAME")        
y2=my_conv2d(y1,3,6,ave,sigma,6,6,1,1,tf.nn.tanh,"SAME")         


y101 = tf.reshape(y2, [tf.shape(x)[0],36*36*6])
y102 = add_layer(y101,36*36*6,36*36*2,tf.nn.tanh)
y103 = add_layer(y102,36*36*2,36*18,tf.nn.tanh)
y104 = add_layer(y103,36*18,36*3,tf.nn.tanh)
y105 = add_layer(y104,36*3,36,tf.nn.tanh)
y106 = add_layer(y105,36,12,tf.nn.tanh)
y107 = add_layer(y106,12,output_dim,None)

print('y1,',y1.shape)
print('y2,',y2.shape)


print('y101,',y101.shape)
print('y102,',y102.shape)
print('y103,',y103.shape)
print('y104,',y104.shape)
print('y105,',y105.shape)
print('y106,',y106.shape)
print('y107,',y107.shape)



#print(tf.reshape(y5[0:batch_size,0,0,0],[batch_size,1,1,1]))#tf.shape(x)[0]
prediction1=tf.reshape(y107[0:tf.shape(x)[0],:],[tf.shape(x)[0],1])
prediction20=tf.convert_to_tensor(tf.gradients(y107,x))
prediction2=tf.reshape(prediction20[0,:,:,:,0],[tf.shape(x)[0],input_dim,input_dim,1])
print('prediction1,',prediction1.shape)
print('prediction20,',prediction20.shape)
print('prediction2,',prediction2.shape)

with tf.name_scope('loss'):
    #loss =tf.reduce_sum(tf.square(y-prediction1))+
    loss1 =tf.reduce_sum(tf.square(y-prediction1)) # or reduce_mean
    loss2 =tf.reduce_sum(tf.square(yg-prediction2)) #or reduce_mean
    loss =tf.reduce_sum(tf.square(y-prediction1))+1e6*tf.reduce_sum(tf.square(yg-prediction2)) #or reduce_mean
    print('loss',loss)

def get_loss(loss):
    #print(loss)
    losses.append(loss)


def SecondOrderOptimizer():
    with tf.name_scope('train'):
        train = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': num_iters})
    
    with tf.Session() as sess:
        #writer = tf.summary.FileWriter("logs/",sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection_ref("p_var"))
        if False:
           saver.restore(sess,'./savemodel2/model')
        train.minimize(sess, fetches=[loss],loss_callback=get_loss,feed_dict={x:x_data,y:y_data,yg:yg_data})
        saver.save(sess, "./savemodel2/model")
        
        prediction_value_Train,prediction_grad_Train = sess.run([prediction1,prediction2],feed_dict={x:x_data})
        plt.figure()
        plt.scatter(y_data[:],prediction_value_Train[:], alpha=0.5)
        plt.plot(y_data[:],y_data[:],'r-')
        #plt.show()
        plt.savefig("pic2/Train_accuracy_c.png")
        plt.close()
        
        plt.figure()
        plt.scatter(y_data[0:dpn],prediction_value_Train[0:dpn], alpha=0.5)
        plt.plot(y_data[0:dpn],y_data[0:dpn],'r-')
        #plt.show()
        plt.savefig("pic2/Train_accuracy_c_part.png")
        plt.close()
        
        plt.figure()
        plt.scatter(yg_data[:].reshape(yg_data.shape[0]*input_dim*input_dim),prediction_grad_Train[:].reshape(prediction_grad_Train.shape[0]*input_dim*input_dim), alpha=0.5)
        plt.plot(yg_data[:].reshape(yg_data.shape[0]*input_dim*input_dim),yg_data[:].reshape(yg_data.shape[0]*input_dim*input_dim),'r-')
        #plt.show()
        plt.savefig("pic2/Train_accuracy_grad.png")
        plt.close()
        

        
        plt.figure()
        plt.scatter(yg_data[0:dpn].reshape(dpn*input_dim*input_dim),prediction_grad_Train[0:dpn].reshape(dpn*input_dim*input_dim), alpha=0.5)
        plt.plot(yg_data[0:dpn].reshape(dpn*input_dim*input_dim),yg_data[0:dpn].reshape(dpn*input_dim*input_dim),'r-')
        #plt.show()
        plt.savefig("pic2/Train_accuracy_grad_part.png")
        plt.close()
        
        np.savetxt('sto2/lossinformation.csv', losses, delimiter=",")
        
        
        prediction_value_Test,prediction_grad_Test = sess.run([prediction1,prediction2],feed_dict={x:x_test})
        np.savetxt('sto2/prediction_Test.csv', prediction_value_Test, delimiter=",")
        plt.figure()
        plt.scatter(y_test[:],prediction_value_Test[:], alpha=0.5)
        plt.plot(y_test[:],y_test[:],'r-')
        #plt.show()
        plt.savefig("pic2/test_accuracy_c.png")
        plt.close()
        print('test_relative_error_c=%f\n',np.mean(np.abs(y_test[:].reshape(y_test.shape[0])-prediction_value_Test[:].reshape(prediction_value_Test.shape[0]))/np.maximum(np.abs(y_test[:].reshape(y_test.shape[0])),2e-1)))
        
        plt.figure()
        plt.scatter(y_test[0:dpn],prediction_value_Test[0:dpn], alpha=0.5)
        plt.plot(y_test[0:dpn],y_test[0:dpn],'r-')
        #plt.show()
        plt.savefig("pic2/test_accuracy_c_part.png")
        plt.close()
        
        plt.figure()
        plt.scatter(yg_test[:].reshape(yg_test.shape[0]*input_dim*input_dim),prediction_grad_Test[:].reshape(prediction_grad_Test.shape[0]*input_dim*input_dim), alpha=0.5)
        plt.plot(yg_test[:].reshape(yg_test.shape[0]*input_dim*input_dim),yg_test[:].reshape(yg_test.shape[0]*input_dim*input_dim),'r-')
        #plt.show()
        plt.savefig("pic2/test_accuracy_grad.png")
        plt.close()

        print('test_relative_error_s=%f\n',np.mean(np.abs(yg_test[:].reshape(yg_test.shape[0]*input_dim*input_dim)-prediction_grad_Test[:].reshape(prediction_grad_Test.shape[0]*input_dim*input_dim))/np.maximum(np.abs(yg_test[:].reshape(yg_test.shape[0]*input_dim*input_dim)),2.5e-2)))

        
        plt.figure()
        plt.scatter(yg_test[0:dpn].reshape(dpn*input_dim*input_dim),prediction_grad_Test[0:dpn].reshape(dpn*input_dim*input_dim), alpha=0.5)
        plt.plot(yg_test[0:dpn].reshape(dpn*input_dim*input_dim),yg_test[0:dpn].reshape(dpn*input_dim*input_dim),'r-')
        #plt.show()
        plt.savefig("pic2/test_accuracy_grad_part.png")
        plt.close()        
        
        prediction_value_valid,prediction_grad_valid = sess.run([prediction1,prediction2],feed_dict={x:x_valid})
        np.savetxt('sto2/prediction_valid.csv', prediction_value_valid, delimiter=",")
        
        #valid_accuracy = np.mean(abs(1-np.abs(prediction_value_valid)/(np.abs(y_valid)+((np.abs(y_valid))<1e-1)*1e-1))<0.05)   
        
        valid_accuracy = np.mean(np.abs(prediction_value_valid-y_valid)<1e-1)  
        print('Valid:acc2 = {0}'.format(valid_accuracy))
        
        plt.figure()
        plt.scatter(y_valid[:],prediction_value_valid[:],alpha=0.5)
        
        plt.plot(y_valid[:],y_valid[:],'r-')
        #plt.show()
        plt.savefig("pic2/valid_accuracy_c.png")
        plt.close()
        
        plt.figure()
        plt.scatter(y_valid[0:dpn],prediction_value_valid[0:dpn],alpha=0.5)
        plt.plot(y_valid[0:dpn],y_valid[0:dpn],'r-')
        
        #plt.show()
        plt.savefig("pic2/valid_accuracy_c_part.png")
        plt.close()
        
        plt.figure()
        plt.scatter(yg_valid[:].reshape(yg_valid.shape[0]*input_dim*input_dim),prediction_grad_valid[:].reshape(prediction_grad_valid.shape[0]*input_dim*input_dim),alpha=0.5)
        plt.plot(yg_valid[:].reshape(yg_valid.shape[0]*input_dim*input_dim),yg_valid[:].reshape(yg_valid.shape[0]*input_dim*input_dim),'r-')
        #plt.show()
        plt.savefig("pic2/valid_accuracy_grad.png")
        plt.close()
        
        plt.figure()
        plt.scatter(yg_valid[0:dpn].reshape(dpn*input_dim*input_dim),prediction_grad_valid[0:dpn].reshape(dpn*input_dim*input_dim),alpha=0.5)
        plt.plot(yg_valid[0:dpn].reshape(dpn*input_dim*input_dim),yg_valid[0:dpn].reshape(dpn*input_dim*input_dim),'r-')
        #plt.show()
        plt.savefig("pic2/valid_accuracy_grad_part.png")
        plt.close()



SecondOrderOptimizer()
losses = []
