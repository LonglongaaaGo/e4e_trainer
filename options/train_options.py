from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='Encoder4Editing', type=str, help='Which encoder to use')

        self.parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')

        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')

        # self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
        #                          help='Path to StyleGAN model weights')

        self.parser.add_argument('--stylegan_size', default=1024, type=int,
                                 help='size of pretrained StyleGAN Generator')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=2000, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=500, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=100, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=1000, type=int, help='Model checkpoint interval')

        # Discriminator flags
        self.parser.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
        self.parser.add_argument('--w_discriminator_lr', default=2e-5, type=float, help='Dw learning rate')
        self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        self.parser.add_argument("--d_reg_every", type=int, default=16,
                                 help="interval for applying r1 regularization")
        self.parser.add_argument('--use_w_pool', action='store_true',
                                 help='Whether to store a latnet codes pool for the discriminator\'s training')
        self.parser.add_argument("--w_pool_size", type=int, default=50,
                                 help="W\'s pool size, depends on --use_w_pool")

        # e4e specific
        self.parser.add_argument('--delta_norm', type=int, default=2, help="norm type of the deltas")
        self.parser.add_argument('--delta_norm_lambda', type=float, default=2e-4, help="lambda for delta norm loss")

        # Progressive training
        self.parser.add_argument('--progressive_steps', nargs='+', type=int, default=None,
                                 help="The training steps of training new deltas. steps[i] starts the delta_i training")
        self.parser.add_argument('--progressive_start', type=int, default=None,
                                 help="The training step to start training the deltas, overrides progressive_steps")
        self.parser.add_argument('--progressive_step_every', type=int, default=2_000,
                                 help="Amount of training steps for each progressive step")

        # Save additional training info to enable future training continuation from produced checkpoints
        self.parser.add_argument('--save_training_data', action='store_true',
                                 help='Save intermediate training data to resume training from the checkpoint')
        self.parser.add_argument('--sub_exp_dir', default=None, type=str, help='Name of sub experiment directory')
        self.parser.add_argument('--keep_optimizer', action='store_true',
                                 help='Whether to continue from the checkpoint\'s optimizer')
        self.parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                 help='Path to training checkpoint, works when --save_training_data was set to True')
        self.parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                 help="Name of training parameters to update the loaded training checkpoint")
 
        ##
        self.parser.add_argument('--stylegan_weights',default="/media/onelong/Longlongaaago_memo/pre-train/e4e/stylegan2-ffhq-config-f.pt",
                                 type=str,help='Path to StyleGAN model weights')
        self.parser.add_argument('--moco_weight_path',default="/media/onelong/Longlongaaago_memo/pre-train/moco_v2_800ep_pretrain.pt",
                                 type=str, help='Path to moco v2 model weights')
        self.parser.add_argument('--shape_predictor_path',
                                 default="/media/onelong/Longlongaaago_memo/pre-train/shape_predictor_68_face_landmarks.dat",
                                 type=str, help='Path to moco v2 model weights')
        self.parser.add_argument('--ir_se50_path',default="/media/onelong/Longlongaaago_memo/pre-train/model_ir_se50.pth",
                                 type=str, help='Path to ir path ')
        ###
        self.parser.add_argument('--img_path',default="/home/iccv/workspace/Data/ffhq_256_anno/train_256",
                                 type=str,help='Path to StyleGAN model weights')
        self.parser.add_argument('--edge_root',default="/home/iccv/workspace/Data/ffhq_sketches/train_256_edges", type=str,
                         help='Path to StyleGAN model weights')
        self.parser.add_argument('--semantic_root',
                                 default="/home/iccv/workspace/Data/ffhq_semantic/train_semantic_mask",
                                 type=str,help='Path to StyleGAN model weights')

        self.parser.add_argument('--color_root',default="/home/iccv/workspace/Data/ffhq_256_super30/train_256",
                                 type=str, help='Path to StyleGAN model weights')

        self.parser.add_argument('--img_test_path',
                                 default="/home/iccv/workspace/Data/ffhq_256_anno/test_256",
                                 type=str, help='Path to StyleGAN model weights')
        self.parser.add_argument('--edge_test_root',
                                 default="/home/iccv/workspace/Data/ffhq_sketches/test_256_edges",
                                 type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--semantic_test_root',
                                 default="/home/iccv/workspace/Data/ffhq_semantic/test_semantic_mask",
                                 type=str, help='Path to StyleGAN model weights')

        self.parser.add_argument('--color_test_root',
                                 default="/home/iccv/workspace/Data/ffhq_256_super30/test_256",
                                 type=str, help='Path to StyleGAN model weights')

        self.parser.add_argument("--canny_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--landmark_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--hed_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--depth_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--midas_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--geometry_path", type=str, help="path to the superpixel_path1 100")

        self.parser.add_argument("--canny_test_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--landmark_test_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--hed_test_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--depth_test_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--midas_test_path", type=str, help="path to the superpixel_path1 100")
        self.parser.add_argument("--geometry_test_path", type=str, help="path to the superpixel_path1 100")

        self.parser.add_argument("--size", type=int, default=256, help="image sizes for the models")
        self.parser.add_argument("--in_channel", type=int, default=19+1+3+3, help="image sizes for the models")

        #False means reuse
        self.parser.add_argument('--multi_modal', default=False, type=bool, help='Whether to reuse the pre-trained parameters from input cnn')

        self.parser.add_argument('--same_channel', default=False, type=bool, help='if or not the input channel number is different')


        self.parser.add_argument('--condition_type', default="image", type=str, help='the type of conditional model')
        self.parser.add_argument('--condition_path', default="/home/iccv/workspace/Data/afhq_v2/test/dog", type=str, help='the type of conditional model')
        self.parser.add_argument('--condition_test_path', default="/home/iccv/workspace/Data/afhq_v2/test/dog", type=str, help='the type of conditional model')





    def parse(self):
        opts = self.parser.parse_args()
        return opts
