import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
from utils import transformer_flow

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(inputs, ref):
	iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
	rH, rW = ref.get_shape()[1], ref.get_shape()[2]
	if iH == rH and iW == rW:
		return inputs
	return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

################## PoseNet #####################
def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True,reuse = tf.AUTO_REUSE):
	inputs = tf.concat([tgt_image, src_image_stack], axis=3)
	num_source = int(src_image_stack.get_shape()[3].value//3)
	with tf.variable_scope('pose_exp_net',reuse = reuse) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
							normalizer_fn=None,
							weights_regularizer=slim.l2_regularizer(0.05),
							activation_fn=tf.nn.relu,
							outputs_collections=end_points_collection):
			# cnv1 to cnv5b are shared between pose and explainability prediction
			cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
			cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
			cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
			cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
			cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
			# Pose specific layers
			with tf.variable_scope('pose'):
				cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
				cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
				pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', 
					stride=1, normalizer_fn=None, activation_fn=None)
				pose_avg = tf.reduce_mean(pose_pred, [1, 2])
				# Empirically we found that scaling by a small constant 
				# facilitates training.
				pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
			# Exp mask specific layers
			if do_exp:
				with tf.variable_scope('exp'):
					upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

					upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
					mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', 
						normalizer_fn=None, activation_fn=None)

					upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
					mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', 
						normalizer_fn=None, activation_fn=None)
					
					upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
					mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', 
						normalizer_fn=None, activation_fn=None)

					upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
					mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', 
						normalizer_fn=None, activation_fn=None)
			else:
				mask1 = None
				mask2 = None
				mask3 = None
				mask4 = None
			end_points = utils.convert_collection_to_dict(end_points_collection)
			return pose_final, [mask1, mask2, mask3, mask4], end_points


######### DepthNet ##############
def disp_net(tgt_image, is_training=True,reuse = tf.AUTO_REUSE):
	H = tgt_image.get_shape()[1].value
	W = tgt_image.get_shape()[2].value
	with tf.variable_scope('depth_net',reuse = reuse) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
							normalizer_fn=None,
							weights_regularizer=slim.l2_regularizer(0.05),
							activation_fn=tf.nn.relu, trainable = is_training,
							outputs_collections=end_points_collection):
			cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
			cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
			cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
			cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
			cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
			cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
			cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
			cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
			cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
			cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
			cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
			cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
			cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
			cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

			upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
			# There might be dimension mismatch due to uneven down/up-sampling
			upcnv7 = resize_like(upcnv7, cnv6b)
			i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
			icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

			upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
			upcnv6 = resize_like(upcnv6, cnv5b)
			i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
			icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

			upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
			upcnv5 = resize_like(upcnv5, cnv4b)
			i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
			icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

			upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
			i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
			icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
			disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
				activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
			disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

			upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
			i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
			icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
			disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
				activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
			disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

			upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
			i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
			icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
			disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
				activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
			disp2_up = tf.image.resize_bilinear(disp2, [H, W])

			upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
			i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
			icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
			disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
				activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
			
			end_points = utils.convert_collection_to_dict(end_points_collection)
			return [disp1, disp2, disp3, disp4], end_points


######################### PWC-Net ###################
#  Original Implemented in https://github.com/baidu-research/UnDepthflow/blob/master/nets/pwc_flow.py
def leaky_relu(x, alpha=0.1):
	return tf.nn.leaky_relu(x,alpha)


def feature_pyramid(image, reuse = tf.AUTO_REUSE, is_training = True):
	with tf.variable_scope('feature_net'):
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
						normalizer_fn = None,
						weights_regularizer=slim.l2_regularizer(0.0004),
						activation_fn = leaky_relu,
						variables_collections=["flownet"],
						reuse=reuse,trainable = is_training):
			cnv1 = slim.conv2d(image, 16, [3, 3], stride=2, scope="cnv1")
			cnv2 = slim.conv2d(cnv1, 16, [3, 3], stride=1, scope="cnv2")
			cnv3 = slim.conv2d(cnv2, 32, [3, 3], stride=2, scope="cnv3")
			cnv4 = slim.conv2d(cnv3, 32, [3, 3], stride=1, scope="cnv4")
			cnv5 = slim.conv2d(cnv4, 64, [3, 3], stride=2, scope="cnv5")
			cnv6 = slim.conv2d(cnv5, 64, [3, 3], stride=1, scope="cnv6")
			cnv7 = slim.conv2d(cnv6, 96, [3, 3], stride=2, scope="cnv7")
			cnv8 = slim.conv2d(cnv7, 96, [3, 3], stride=1, scope="cnv8")
			cnv9 = slim.conv2d(cnv8, 128, [3, 3], stride=2, scope="cnv9")
			cnv10 = slim.conv2d(cnv9, 128, [3, 3], stride=1, scope="cnv10")
			cnv11 = slim.conv2d(cnv10, 192, [3, 3], stride=2, scope="cnv11")
			cnv12 = slim.conv2d(cnv11, 192, [3, 3], stride=1, scope="cnv12")
		
			return cnv2, cnv4, cnv6, cnv8, cnv10, cnv12 

def cost_volumn(feature1, feature2, d=4):
	batch_size, H, W, feature_num = map(int, feature1.get_shape()[0:4])  
	feature2 = tf.pad(feature2, [[0,0], [d,d], [d,d],[0,0]], "CONSTANT")
	cv = []
	for i in range(2*d+1):
		for j in range(2*d+1):
			cv.append(tf.reduce_mean(feature1*feature2[:, i:(i+H), j:(j+W), :], axis=3, keepdims=True))

	return tf.concat(cv, axis=3)

def optical_flow_decoder_dc(inputs, level, is_training = True):
	with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
					  normalizer_fn = None,
					  weights_regularizer=slim.l2_regularizer(0.0004),
					  activation_fn=leaky_relu,
					  trainable = is_training):
		cnv1 = slim.conv2d(inputs, 128, [3, 3], stride=1, scope="cnv1_fd_"+str(level))
		cnv2 = slim.conv2d(cnv1, 128, [3, 3], stride=1, scope="cnv2_fd_"+str(level))
		cnv3 = slim.conv2d(tf.concat([cnv1, cnv2], axis=3), 96, [3, 3], stride=1, scope="cnv3_fd_"+str(level))
		cnv4 = slim.conv2d(tf.concat([cnv2, cnv3], axis=3), 64, [3, 3], stride=1, scope="cnv4_fd_"+str(level))
		cnv5 = slim.conv2d(tf.concat([cnv3, cnv4], axis=3), 32, [3, 3], stride=1, scope="cnv5_fd_"+str(level))
		flow = slim.conv2d(tf.concat([cnv4, cnv5], axis=3), 2, [3, 3], stride=1, scope="cnv6_fd_"+str(level), activation_fn=None)

		return flow, cnv5
	
def context_net(inputs, is_training = True):
	with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
					  normalizer_fn = None,
					  weights_regularizer  = slim.l2_regularizer(0.0004),
					  trainable = is_training,
					  activation_fn = leaky_relu):
		cnv1 = slim.conv2d(inputs, 128, [3, 3], rate=1, scope="cnv1_cn")
		cnv2 = slim.conv2d(cnv1, 128, [3, 3], rate=2, scope="cnv2_cn")
		cnv3 = slim.conv2d(cnv2, 128, [3, 3], rate=4, scope="cnv3_cn")
		cnv4 = slim.conv2d(cnv3, 96, [3, 3], rate=8, scope="cnv4_cn")
		cnv5 = slim.conv2d(cnv4, 64, [3, 3], rate=16, scope="cnv5_cn")
		cnv6 = slim.conv2d(cnv5, 32, [3, 3], rate=1, scope="cnv6_cn")

		flow = slim.conv2d(cnv6, 2, [3, 3], rate=1, scope="cnv7_cn", activation_fn=None)

		return flow
	
def construct_model_pwc_full(image1, image2, feature1, feature2, is_training = True,reuse = tf.AUTO_REUSE):
	with tf.variable_scope('flow_net',reuse = reuse):
		batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])  
		

		feature1_1, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = feature1
		feature2_1, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = feature2
		
		cv6 = cost_volumn(feature1_6, feature2_6, d=4)
		flow6, _ = optical_flow_decoder_dc(cv6, level=6,is_training = is_training)
		
		flow6to5 = tf.image.resize_bilinear(flow6, [int(H/(2**5)), int((W/(2**5)))]) * 2.0
		feature2_5w = transformer_flow(feature2_5, flow6to5, [int(H/(2**5)), int((W/(2**5)))])
		cv5 = cost_volumn(feature1_5, feature2_5w, d=4)
		flow5, _ = optical_flow_decoder_dc(tf.concat([cv5, feature1_5, flow6to5], axis=3), level=5, is_training = is_training) 
		flow5 = flow5 + flow6to5
		
		flow5to4 = tf.image.resize_bilinear(flow5, [int(H/(2**4)), int((W/(2**4)))]) * 2.0
		feature2_4w = transformer_flow(feature2_4, flow5to4, [int(H/(2**4)), int((W/(2**4)))])
		cv4 = cost_volumn(feature1_4, feature2_4w, d=4)
		flow4, _ = optical_flow_decoder_dc(tf.concat([cv4, feature1_4, flow5to4], axis=3), level=4, is_training = is_training)
		flow4 = flow4 + flow5to4
		
		flow4to3 = tf.image.resize_bilinear(flow4, [int(H/(2**3)), int((W/(2**3)))]) * 2.0
		feature2_3w = transformer_flow(feature2_3, flow4to3, [int(H/(2**3)), int((W/(2**3)))])
		cv3 = cost_volumn(feature1_3, feature2_3w, d=4)
		flow3, _ = optical_flow_decoder_dc(tf.concat([cv3, feature1_3, flow4to3], axis=3), level=3, is_training = is_training)
		flow3 = flow3 + flow4to3
		
		flow3to2 = tf.image.resize_bilinear(flow3, [int(H/(2**2)), int((W/(2**2)))]) * 2.0
		feature2_2w = transformer_flow(feature2_2, flow3to2, [int(H/(2**2)), int((W/(2**2)))])
		cv2 = cost_volumn(feature1_2, feature2_2w, d=4)
		flow2_raw, f2 = optical_flow_decoder_dc(tf.concat([cv2, feature1_2, flow3to2], axis=3), level=2, is_training = is_training) 
		flow2_raw = flow2_raw + flow3to2
		
		flow2 = context_net(tf.concat([flow2_raw, f2], axis=3),is_training = is_training) + flow2_raw
		
		flow0_enlarge = tf.image.resize_bilinear(flow2*4.0, [H, W])
		flow1_enlarge = tf.image.resize_bilinear(flow3*4.0, [int(H/(2**1)), int((W/(2**1)))])
		flow2_enlarge = tf.image.resize_bilinear(flow4*4.0, [int(H/(2**2)), int((W/(2**2)))])
		flow3_enlarge = tf.image.resize_bilinear(flow5*4.0, [int(H/(2**3)), int((W/(2**3)))])
		
		return flow0_enlarge, flow1_enlarge, flow2_enlarge, flow3_enlarge