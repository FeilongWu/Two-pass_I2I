import time
from options.train_options import TrainOptions
from data import create_dataset
from data import load_y
from models import create_model
from util.visualizer import Visualizer
import copy

if __name__ == '__main__':
##    opt = TrainOptions().parse()
##    dataset = create_dataset(opt)
##    dataset_size = len(dataset)
##    print('#training images = %d' % dataset_size)

    #opt1 = copy.deepcopy(opt)
    #dataset_mode = 'alignedpseudo'
    #opt1.dataset_mode = dataset_mode
    
    opt = TrainOptions().parse()
    #opt.preprocess = 'resize'
    opt.dataset_mode = 'alignedpseudo'
    opt.gan_mode = 'vanilla'
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
    opt1 = copy.deepcopy(opt)
    opt1.dataroot = opt1.pseudo_data_root
    #######
    opt1.serial_batches = True
    #opt1.preprocess = 'resize'
    opt1.dataset_mode = 'alignedpseudo1'
    #######
    dataset_mode = 'alignedpseudo'
    dataset_pseudo = create_dataset(opt1)
    opt1.dataset_mode = 'alignedpseudo'
    dataset_size_pseudo = len(dataset_pseudo)
    print('#training images = %d' % dataset_size_pseudo)

    ### load pseudo label with numpy, which is updated along with network parameters

##    for ii in range(1):
##        f=open('aligned.txt','a')
##        for i,data in enumerate(dataset):
##            f.write(str(data)+'\n')
##            if i==3:
##                break
##        f.close()
##        f=open('pseudo_aligned.txt','a')
##        for j,data in enumerate(dataset_pseudo):
##            f.write(str(data)+'\n')
##            if j==3:
##                break
##        f.close()


        ####
    # intermediate y create or load
    y_i = load_y(opt1,dataset_pseudo)

    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    decay = False

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if epoch > opt.n_epochs:
            decay = True
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        for i, data in enumerate(dataset_pseudo):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            data['B'] = y_i[i] # replace pseudo label with the y intermediate
            #data['B'].grad = None
            model.set_input(data, decay = decay, dataset_mode = dataset_mode) # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                #print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

