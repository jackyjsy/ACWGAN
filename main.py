import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Data loader
    data_loader = None

    if config.dataset == 'CelebA':
        image_path = os.path.join(config.celebA_path, 'images')
        seg_path = os.path.join(config.celebA_path, 'segmentation')
        metadata_path = os.path.join(config.celebA_path, 'list_attr_celeba_s.txt')

        data_loader = get_loader(image_path, seg_path, metadata_path, config.celebA_crop_size,
                            config.image_size, config.batch_size, 'CelebA', config.mode)
    elif config.dataset == 'Fashion':
        image_path = os.path.join(config.fashion_path, 'Img')
        seg_path = os.path.join(config.fashion_path, 'Img')
        metadata_path = os.path.join(config.fashion_path, 'Anno')
        data_loader = get_loader(image_path, seg_path, metadata_path, config.fashion_crop_size,
                            config.image_size, config.batch_size, 'Fashion', config.mode)
        config.c_dim = 17
    # Solver
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        if config.dataset == 'CelebA':
            solver.test_celeba_single()
        if config.dataset == 'Fashion':
            solver.test()
    elif config.mode == 'test_seg':
        solver.test_seg()
    elif config.mode == 'test_interp':
        solver.test_interp()
    elif config.mode == 'test_interp_all':
        solver.test_interp_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--s_dim', type=int, default=7)
    parser.add_argument('--celebA_crop_size', type=int, default=178)
    parser.add_argument('--fashion_crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--d_lr', type=float, default=0.0002)
    parser.add_argument('--a_lr', type=float, default=0.0002)
    parser.add_argument('--lambda_cls', type=float, default=5)
    parser.add_argument('--lambda_s', type=float, default=10)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'Fashion'])
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Test settings
    parser.add_argument('--test_model', type=str, default='20_1000')
    parser.add_argument('--test_seg_path', type=str, default='data/CelebA_nocrop/test/nosmile2smile/')
    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_seg', 'test_interp', 'test_interp_all'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    # parser.add_argument('--celebA_image_path', type=str, default='./data/CelebA_nocrop/images')
    # parser.add_argument('--celebA_seg_path', type=str, default='./data/CelebA_nocrop/Segmentation')
    # parser.add_argument('--metadata_path', type=str, default='./data/list_attr_celeba_s.txt')
    parser.add_argument('--celebA_path', type=str, default='/home/songyao/workspace/GAN/pytorch-current/data/CelebA_nocrop')
    parser.add_argument('--fashion_path', type=str, default='/home/songyao/workspace/data/DeepFashion')

    # parser.add_argument('--log_path', type=str, default='./experiment/logs')
    # parser.add_argument('--model_save_path', type=str, default='./experiment/models')
    # parser.add_argument('--sample_path', type=str, default='./experiment/samples')
    # parser.add_argument('--result_path', type=str, default='./experiment/results')
    parser.add_argument('--experiment_path', type=str, default='./experiment')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=1000)
    
    # Visdom setting
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--web_dir', type=str, default='web')

    config = parser.parse_args()
    config.log_path = os.path.join(config.experiment_path, 'logs')
    config.model_save_path = os.path.join(config.experiment_path, 'models')
    config.sample_path = os.path.join(config.experiment_path, 'samples')
    config.result_path = os.path.join(config.experiment_path, 'results')

    print(config)
    main(config)