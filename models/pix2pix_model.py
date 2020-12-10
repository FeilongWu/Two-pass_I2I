import torch
from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks
from .networks import MedianPool2d


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Lc', type=float, default=15.0, help='weight for classification loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake'] # Lc = classification loss
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            use_sigmoid = True
            #self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_sigmoid = use_sigmoid) # for relativistic loss

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #self.optimizers.append(self.optimizer_D2)
            self.im_pool = ImagePool(opt.pool_size)
            #self.MedianPool = MedianPool2d()
            #self.softmax = torch.nn.Softmax(dim=1)
            #self.logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()

    def set_input(self, input, decay = False, dataset_mode = 'aligned'):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.decay = decay
        self.dataset_mode = dataset_mode
        
        AtoB = self.opt.direction == 'AtoB'
        
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.GT = input['ground_truth'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        

##    def backward_D(self):
##        """Calculate GAN loss for the discriminator"""
##        # Fake; stop backprop to the generator by detaching fake_B
##        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
##        pred_fake = self.netD(fake_AB.detach())
##        self.loss_D_fake = self.criterionGAN(pred_fake, False)
##        # Real
##        real_AB = torch.cat((self.real_A, self.real_B), 1)
##        pred_real = self.netD(real_AB)
##        self.loss_D_real = self.criterionGAN(pred_real, True)
##        # combine loss and calculate gradients
##        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
##        self.loss_D.backward()

    def backward_D(self):
        fake_B = self.im_pool.query(self.fake_B) # prediction
        self.loss_D = self.backward_D_basic(self.netD, self.GT, fake_B)
        
    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
       # if(self.opt.loss_case=='rel_bce'):
        loss_D = self.criterionGAN(pred_real - pred_fake, True)
       # else:
       #     loss_D = (torch.mean((pred_real - torch.mean(pred_fake) - 1.0) ** 2) + torch.mean((pred_fake - torch.mean(pred_real) + 1.0) ** 2))/2
        loss_D.backward()
        self.loss_D_real = loss_D
        self.loss_D_fake = -99
        return loss_D

##    def backward_D2(self):
##        # calculate relativistic loss
##        fake_B = self.im_pool.query(self.fake_B) # prediction
##        if self.dataset_mode == 'alignedpseudo':
##            self.loss_D2 = self.backward_D_basic(self.netD2, self.GT, fake_B)
##        else:
##            self.loss_D2 = self.backward_D_basic(self.netD2, self.real_B, fake_B)

    def L_classification(self):
        # calculate classification loss
        gt_b = self.MedianPool(self.real_B)
        pre_b = self.MedianPool(self.fake_B)

        loss = torch.mean(torch.mul(pre_b, torch.log(torch.div(self.softmax(pre_b),self.softmax(gt_b)))))
        return loss

    def entropy(self):
        # calculate entropy loss
        pre_B = self.MedianPool(self.fake_B)
        loss = - torch.mean(torch.mul(self.softmax(pre_B), self.logsoftmax(pre_B)))
        return loss


    def backward_G(self):
        # identity loss use only for pseudo labels
        if self.dataset_mode == 'alignedpseudo':
            tot_lambda = 0.1
            pass
        else:
            tot_lambda = 1
            self.idt_loss = 0

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        pred_fake_rel = self.netD(self.im_pool.query(self.fake_B)) # prediction for relativistic loss
        pred_real_rel = self.netD(self.real_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake_rel - pred_real_rel, True)
        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 + self.idt_loss) * tot_lambda
        self.loss_G.backward()
        

##    def backward_G(self):
##        """Calculate GAN and L1 loss for the generator"""
##        pred_fake_rel = self.netD2(self.im_pool.query(self.fake_B)) # prediction for relativistic loss
##        
##        if self.dataset_mode != 'alignedpseudo':
##            # First, G(A) should fake the discriminator
##            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
##            pred_fake = self.netD(fake_AB)
##            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
##            
##            self.classification_loss = self.L_classification()
##            
##            pred_real_rel = self.netD2(self.real_B)
##            
##        else:
##            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
##            pred_fake = self.netD(fake_AB)
##            pro_fake = self.criterionGAN(pred_fake, False)
##            real_AB = torch.cat((self.real_A, self.real_B), 1)
##            pred_real = self.netD(real_AB)
##            pro_real = self.criterionGAN(pred_real, False)
##            self.loss_G_GAN = pro_real / pro_fake
##            
##            self.classification_loss = 0
##            
##            pred_real_rel = self.netD2(self.GT)
##        #self.loss_G_GAN2 = (torch.mean((pred_real - torch.mean(pred_fake) + 1.0)\
##        #                                   ** 2) + torch.mean((pred_fake - torch.mean(pred_real) - 1.0) ** 2))/2
##        self.loss_G_GAN2 = self.criterionGAN(pred_fake_rel - pred_real_rel, True)
##        self.entropy_L = self.entropy()
##        self.loss_G_GAN += self.loss_G_GAN2
##        # Second, G(A) = B
##        self.loss_G_L1 = 0
##        if not self.decay:
##            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
##        # combine loss and calculate gradients
##        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.entropy_L + self.classification_loss * self.opt.lambda_Lc
##        self.loss_G.backward()

##        #########################
##        url='lossGAN.txt'
##        url1='lossL1.txt'
##        f=open(url,'a')
##        f.write(str(self.loss_G_GAN))
##        f.close()
##        f=open(url1,'a')
##        f.write(str(self.loss_G_L1))
##        f.close()
##        #########################

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
##        if self.dataset_mode != 'alignedpseudo':
##            self.set_requires_grad(self.netD, True)  # enable backprop for D
##            self.optimizer_D.zero_grad()     # set D's gradients to zero
##            self.backward_D()                # calculate gradients for D
##            self.optimizer_D.step()          # update D's weights
##        
##        self.set_requires_grad(self.netD2, True)
##        self.optimizer_D2.zero_grad()
##        self.backward_D2()
##        self.optimizer_D2.step()

        self.set_requires_grad(self.netD, True)    # enable backprop for D
        self.optimizer_D.zero_grad()            # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
