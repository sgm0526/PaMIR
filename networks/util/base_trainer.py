from __future__ import division
import sys
import time
import numpy as np
import torch
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .data_loader import CheckpointDataLoader
from .saver import CheckpointSaver
from .util import configure_logging

from torch.utils.data import DataLoader
from evaluator_tex import EvaluatorTex

from torchvision.utils import save_image

tqdm.monitor_interval = 0


class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        configure_logging(self.options.debug, self.options.quiet, self.options.logfile)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.dataset_perm = None
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(
                self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model])
                    logging.info('Checkpoint loaded')

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume
            # from an arbitrary step inside an epoch
            def worker_init_fn(worker_id):  # set numpy's random seed
                seed = torch.initial_seed()
                seed = seed % (2 ** 32)
                np.random.seed(seed + worker_id)

            self.start_epoch(epoch)
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     dataset_perm=self.dataset_perm,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train,
                                                     worker_init_fn=worker_init_fn)

            val_data_loader = DataLoader(self.val_ds,batch_size=1, shuffle=False, num_workers=self.options.num_workers,
                worker_init_fn=worker_init_fn, drop_last=False)



            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:

                        self.train_summaries(batch, out)
                    if self.step_count % (5*self.options.summary_steps) == 0:
                        self.summary_writer.add_images('source_img', batch['img'], self.step_count)
                        self.summary_writer.add_images('target_img', batch['target_img'], self.step_count)
                        # self.summary_writer.add_images('nerf_img', pixels_high, self.step_count)
                        # self.summary_writer.add_images('down_nerf_img', pred_img, self.step_count)

                    if self.step_count % (50*self.options.summary_steps) == 0:
                        evaluater = EvaluatorTex(self.device, None, None, no_weight=True)
                        evaluater.pamir_net = self.pamir_net
                        evaluater.pamir_tex_net = self.pamir_tex_net
                        for step_val, batch_val in enumerate(tqdm(val_data_loader, desc='Epoch ' + str(epoch),
                                                                  total=len(self.val_ds),
                                                                  initial=0)):
                            if step_val ==1:
                                break
                            batch_val = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                                         batch_val.items()}

                            nerf_color = evaluater.test_nerf_target(batch_val['img'], batch_val['betas'],
                                                                    batch_val['pose'], batch_val['scale'],
                                                                    batch_val['trans'],
                                                                    batch_val['view_id'] - batch_val['target_view_id'])
                            self.summary_writer.add_images('nerf_img', nerf_color , self.step_count)
                            self.summary_writer.add_images('target_img', batch_val['target_img'], self.step_count)


                    if False: #self.step_count % (100*self.options.summary_steps) == 0:
                        #self.val_summaries(batch, out)
                        evaluater = EvaluatorTex(self.device, None, None, no_weight=True)
                        evaluater.pamir_net = self.pamir_net
                        evaluater.pamir_tex_net = self.pamir_tex_net

                        val_mesh_loss = 0
                        val_nerf_loss = 0
                        for step_val , batch_val in enumerate(tqdm(val_data_loader, desc='Epoch ' + str(epoch),
                                                          total=len(self.val_ds),
                                                          initial=0)):

                            batch_val = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch_val.items()}

                            mesh_color = evaluater.test_tex_pifu(batch_val['img'], batch_val['pts'], batch_val['betas'],
                                                    batch_val['pose'], batch_val['scale'], batch_val['trans'])

                            nerf_color = evaluater.test_nerf_target(batch_val['img'], batch_val['betas'],
                                                                    batch_val['pose'], batch_val['scale'], batch_val['trans'],
                                                                    batch_val['view_id'] - batch_val['target_view_id'])

                            val_mesh_loss +=self.tex_loss(batch_val['pts_clr'], torch.Tensor(mesh_color).unsqueeze(0).cuda())
                            val_nerf_loss += self.tex_loss(batch_val['target_img'],
                                                           nerf_color.cuda())
                            #save_image(batch_val['target_img'],f'./debug/{step_val}_1.png')
                        val_mesh_loss /= len(self.val_ds)
                        val_nerf_loss /= len(self.val_ds)
                        self.summary_writer.add_scalar('val_tex', val_mesh_loss.item(), self.step_count)
                        self.summary_writer.add_scalar('val_nerf_tex', val_nerf_loss.item(), self.step_count)

                    # Backup the current training stage
                    if self.step_count % (self.options.summary_steps*10) == 0:
                        self.saver.save_latest(
                            self.models_dict, self.optimizers_dict, epoch, step + 1,
                            self.options.batch_size, train_data_loader.sampler.dataset_perm,
                            self.step_count)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(
                            self.models_dict, self.optimizers_dict, epoch, step+1,
                            self.options.batch_size, train_data_loader.sampler.dataset_perm,
                            self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(
                        self.models_dict, self.optimizers_dict, epoch, step,
                        self.options.batch_size, train_data_loader.sampler.dataset_perm,
                        self.step_count)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(
                # self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(
                    self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size,
                    None, self.step_count)
        return

    # The following methods (with the possible exception of test)
    # have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def start_epoch(self, epoch):
        pass

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        pass
