dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': '/home/k/Workspace/Data/celeba-256/train',
	# 'ffhq': "/home/k/Data/celeba_1024_test",

	# 'ffhq': '/media/k/Longlongaaago4/Dataset/FFHQ_wm/ffhq_1024_lmdb/train_1024_lmdb',

	# 'celeba_test': '/media/k/Longlongaaago4/Dataset/FFHQ_wm/ffhq_1024_lmdb/test_1024_lmdb',
	'celeba_test': '/home/k/Data/celeba_1024_test',

	#  Cars Dataset (In the paper: Stanford cars)
	'cars_train': '',
	'cars_test': '',

	#  Horse Dataset (In the paper: LSUN Horse)
	'horse_train': '',
	'horse_test': '',

	#  Church Dataset (In the paper: LSUN Church)
	'church_train': '',
	'church_test': '',

	#  Cats Dataset (In the paper: LSUN Cat)
	'cats_train': '',
	'cats_test': ''
}

model_paths = {
	'stylegan_ffhq': '/home/k/Workspace/Pre-trained/stylegan2-ffhq-config-f.pt',
	'ir_se50': '/home/k/Workspace/Pre-trained/model_ir_se50.pth',
	'shape_predictor': '/home/k/Workspace/Pre-trained/shape_predictor_68_face_landmarks.dat',
	'moco': '/home/k/Workspace/Pre-trained/moco_v2_800ep_pretrain.pt'
}
