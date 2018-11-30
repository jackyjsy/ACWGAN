import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Discriminator
from model import Generator_CNN
from model import Discriminator_CNN
from PIL import Image
from util.visualizer import Visualizer
import util.util as util
from collections import OrderedDict


class Solver(object):

    def __init__(self, celebA_loader, config):
        # Data loader
        self.celebA_loader = celebA_loader
        self.visualizer = Visualizer(port = config.port, web_dir = config.web_dir)
        # Model hyper-parameters
        self.z_dim = config.z_dim
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model
        self.config = config

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.visual_step = self.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define a generator and a discriminator
        self.G = Generator(self.z_dim, self.c_dim)
        self.D = Discriminator_CNN(self.c_dim) 
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        # self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
        
    def make_celeb_labels_test(self):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,1,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,1,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,1,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,1,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,1,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,1,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,0,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,0,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,0,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,0,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,0,0]).unsqueeze(0), volatile=True))

        return fixed_c_list

    def train(self):
        """Train StarGAN within a single dataset."""

        # Set dataloader
        
        self.data_loader = self.celebA_loader
        

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        # Fixed latent vector and label for output samples
        fixed_size = 20
        fixed_z = torch.randn(fixed_size, self.z_dim)
        fixed_z = self.to_var(fixed_z, volatile=True)

        fixed_c_list = self.make_celeb_labels_test()

        fixed_z_repeat = fixed_z.repeat(len(fixed_c_list),1)
        fixed_c_repeat_list = []
        for fixed_c in fixed_c_list:
            fixed_c_repeat_list.append(fixed_c.expand(fixed_size,fixed_c.size(1)))
        fixed_c_list = []
        fixed_c_repeat = torch.cat(fixed_c_repeat_list, dim=0)
        fixed_c_repeat_list = []
        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])-1
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            epoch_iter = 0
            for i, (real_x, real_label) in enumerate(self.data_loader):
                epoch_iter = epoch_iter + 1
                if self.dataset == 'Fashion':
                    real_c_i = real_label_i.clone()
                real_c = real_label.clone()
                # rand_idx = torch.randperm(real_c.size(0))
                # fake_c = real_c[rand_idx]
                
                z = torch.randn(real_x.size(0), self.z_dim)
                z = self.to_var(z)
                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)           # input for the generator
                if self.dataset == 'Fashion':
                    real_c_i = self.to_var(real_c_i) 
                # fake_c = self.to_var(fake_c, volatile=True)   
                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                d_loss_real = - torch.mean(out_src)
                # print(real_x.size())
                # print(out_src.size())
                # print(out_cls.size())
                # print(real_c.size())
                if self.dataset == 'CelebA':
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_c, size_average=False) / real_x.size(0)
                elif self.dataset == 'Fashion':
                    d_loss_cls = F.cross_entropy(out_cls, real_c_i)

                # # Compute classification accuracy of the discriminator
                # if (i+1) % self.log_step == 0:
                #     accuracies = self.compute_accuracy(out_cls, real_c, self.dataset)
                #     log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                #     if self.dataset == 'CelebA':
                #         print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                #     else:
                #         print('Classification Acc (8 emotional expressions): ', end='')
                #     print(log)

                # Compute loss with fake images
                fake_x = self.G(z, real_c)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(z, real_c)
                    # fake_x2 = self.G(z, fake_c)
                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    if self.dataset == 'CelebA':
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls, real_c, size_average=False) / fake_x.size(0)
                    elif self.dataset == 'Fashion':
                        g_loss_cls = F.cross_entropy(out_cls, real_c_i)
                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]
                
                if (i+1) % self.visual_step == 0:
                    # save visuals
                    self.real_x = real_x
                    self.fake_x = fake_x
                    # self.fake_x2 = fake_x2
                    
                    # save losses
                    self.d_real = - d_loss_real
                    self.d_fake = d_loss_fake
                    self.d_loss = d_loss
                    self.g_loss = g_loss
                    self.g_loss_fake = g_loss_fake
                    self.g_loss_cls = self.lambda_cls * g_loss_cls
                    self.d_loss_cls = self.lambda_cls * d_loss_cls
                    errors_D = self.get_current_errors('D')
                    errors_G = self.get_current_errors('G')
                    self.visualizer.display_current_results(self.get_current_visuals(), e)
                    self.visualizer.plot_current_errors_D(e, float(epoch_iter)/float(iters_per_epoch), errors_D)
                    self.visualizer.plot_current_errors_G(e, float(epoch_iter)/float(iters_per_epoch), errors_G)
                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
#                 if (i+1) % self.sample_step == 0:
# #                     fake_image_list = []
# #                     for fixed_c in fixed_c_list:
# #                         fixed_c = fixed_c.expand(fixed_z.size(0), fixed_c.size(1))
# #                         fake_image_list.append(self.G(fixed_z, fixed_c))
                    
                        
# #                     fake_images = torch.cat(fake_image_list, dim=3)
# #                     save_image(self.denorm(fake_images.data),
# #                         os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
# #                     print('Translated images and saved into {}..!'.format(self.sample_path))

#                     fake_images_repeat = self.G(fixed_z_repeat, fixed_c_repeat)
#                     fake_image_list = []
#                     for idx in range(12):
#                         fake_image_list.append(fake_images_repeat[fixed_size*(idx):fixed_size*(idx+1)])
#                     fake_images = torch.cat(fake_image_list, dim=3)
#                     save_image(self.denorm(fake_images.data),
#                         os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
#                     print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
      
    # def test(self):
    #     test_size = 30
    #     c_dim = 17
    #     test_c = self.to_var(torch.FloatTensor(np.eye(c_dim, dtype=float)), volatile=True)
    #     fake_image_list = []
    #     for i in range(test_size):
    #         test_z = self.to_var(torch.randn(c_dim, self.z_dim), volatile=True)
    #         test_z = test_z.expand(c_dim, test_z.size(1))
    #         fake_image_list.append(self.G(test_z, test_c))

    #     fake_images = torch.cat(fake_image_list, dim=3)
    #     save_image(self.denorm(fake_images.data),
    #                     os.path.join(self.result_path, 'fake.png'),nrow=1, padding=0)
    def make_celeb_labels(self):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []
        fixed_c_list.append(torch.FloatTensor([1,0,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,0]).unsqueeze(0))
        fixed_c = torch.cat(fixed_c_list,dim=0)
        return fixed_c
    def make_celeb_labels_all(self):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []

        fixed_c_list.append(torch.FloatTensor([1,0,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,1,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,1,1]).unsqueeze(0))
        
        fixed_c_list.append(torch.FloatTensor([1,0,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,1,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,1,0]).unsqueeze(0))
        
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,0,1]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,0,1]).unsqueeze(0))
        
        
        
        fixed_c_list.append(torch.FloatTensor([1,0,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,1,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,1,0,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,1,1,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([1,0,1,0,0]).unsqueeze(0))
        fixed_c_list.append(torch.FloatTensor([0,0,0,0,0]).unsqueeze(0))

        return fixed_c_list
    def test_celeba(self):
        test_size = 16
        test_c = self.make_celeb_labels();
        # print(test_c)
        test_c = test_c.repeat(test_size,1)
        test_c = self.to_var(test_c, volatile=True)
        test_z_list = []
        for i in range(16):
            test_z_list.append(torch.randn(1, self.z_dim).repeat(12, 1))
        test_z = torch.cat(test_z_list, dim=0)
        test_z = self.to_var(test_z, volatile=True)
        fake_image_mat = self.G(test_z, test_c)
        # fake_image_save = fake_image_mat.view(16, 3, 128*12, 128)
        save_image(self.denorm(fake_image_mat.data),
                        os.path.join(self.result_path, 'fake.png'),nrow=1, padding=0)
    def test_celeba_single(self):
        image_index = 0
        import math
        test_size = math.ceil(50000/28)
        c_dim = 28
        test_c = self.make_celeb_labels_all()
        test_c = self.to_var(torch.cat(test_c,dim=0), volatile=True)
        for i in range(test_size):
            test_z = self.to_var(torch.randn(c_dim, self.z_dim), volatile=True)
            fake_image_list = self.G(test_z, test_c)
            for ind in range(fake_image_list.size(0)):
                save_image(self.denorm(fake_image_list[ind].data),
                        os.path.join(self.result_path, 'single/fake_{0:05d}.png'.format(image_index)),nrow=1, padding=0)
                image_index = image_index + 1
            if i > test_size-1:
                break
    def test(self):
        import math
        test_size = math.ceil(50000/17)
        c_dim = 17
        test_c = self.to_var(torch.FloatTensor(np.eye(c_dim, dtype=float)), volatile=True)
        fake_image_list = []
        image_index = 0
        for i in range(test_size):
            test_z = self.to_var(torch.randn(c_dim, self.z_dim), volatile=True)
            test_z = test_z.expand(c_dim, test_z.size(1))
            fake_image_list = self.G(test_z, test_c).transpose(2,3)
            for ind in range(fake_image_list.size(0)):
                save_image(self.denorm(fake_image_list[ind].data),
                        os.path.join(self.result_path, 'single/fake_{0:05d}.png'.format(image_index)),nrow=1, padding=0)
                image_index = image_index + 1


        # fake_images = torch.cat(fake_image_list, dim=3)
        # save_image(self.denorm(fake_images.data),
        #                 os.path.join(self.result_path, 'fake.png'),nrow=1, padding=0)
        
        
    def get_current_errors(self, label='all'):
        D_fake = self.d_fake.data[0]
        D_real = self.d_real.data[0]
        D_loss_cls = self.d_loss_cls.data[0]
        D_loss = self.d_loss.data[0]
        G_loss = self.g_loss.data[0]
        G_loss_cls = self.g_loss_cls.data[0]
        G_loss_fake = self.g_loss_fake.data[0]
        if label == 'all':
            return OrderedDict([('D_fake', D_fake), 
                                ('D_real', D_real), 
                                ('D_loss', D_loss),
                                ('G_loss', G_loss), 
                                ('G_loss_fake', G_loss_fake)])
        if label == 'D':
            return OrderedDict([('D_fake', D_fake), 
                                ('D_loss_cls', D_loss_cls),
                                ('D_real', D_real), 
                                ('D_loss', D_loss)
                                ])
        if label == 'G':
            return OrderedDict([('G_loss', G_loss), 
                                ('G_loss_cls', G_loss_cls), 
                                ('G_loss_fake', G_loss_fake)])

    def get_current_visuals(self):
        real_x = util.tensor2im(self.real_x.data)
        fake_x = util.tensor2im(self.fake_x.data)
        # fake_x2 = util.tensor2im(self.fake_x2.data)
        return OrderedDict([('real_x', real_x), 
                            ('fake_x', fake_x), 
                            # ('fake_x2', fake_x2), 
                            ])
            

