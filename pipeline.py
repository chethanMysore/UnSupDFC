# !/usr/bin/env python
"""

"""

import torch
import torch.utils.data
import torchio as tio
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from evaluation.metrics import (IOU, Dice)
from utils.results_analyser import *
from utils.vessel_utils import (load_model, load_model_with_amp, save_model, write_epoch_summary)

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class Pipeline:

    def __init__(self, cmd_args, model, logger, dir_path, checkpoint_path, writer_training, writer_validating):

        self.model = model
        self.logger = logger
        self.learning_rate = cmd_args.learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
        self.num_epochs = cmd_args.num_epochs
        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.CHECKPOINT_PATH = checkpoint_path
        self.DATASET_PATH = dir_path
        self.OUTPUT_PATH = cmd_args.output_path

        self.model_name = cmd_args.model_name
        self.clip_grads = cmd_args.clip_grads
        self.with_apex = cmd_args.apex
        self.num_classes = cmd_args.num_classes

        # image input parameters
        self.patch_size = cmd_args.patch_size
        self.stride_depth = cmd_args.stride_depth
        self.stride_length = cmd_args.stride_length
        self.stride_width = cmd_args.stride_width
        self.samples_per_epoch = cmd_args.samples_per_epoch

        # execution configs
        self.batch_size = cmd_args.batch_size
        self.num_worker = cmd_args.num_worker

        # Losses
        self.sim_loss_coeff = cmd_args.sim_loss_coeff
        self.cont_loss_coeff = cmd_args.cont_loss_coeff
        self.similarity_loss = torch.nn.CrossEntropyLoss()
        self.continuity_loss = torch.nn.L1Loss()

        # Following metrics can be used to evaluate
        self.dice = Dice()
        # self.focalTverskyLoss = FocalTverskyLoss()
        self.iou = IOU()

        self.LOWEST_LOSS = float('inf')

        if self.with_apex:
            self.scaler = GradScaler()

        self.logger.info("Model Hyper Params: ")
        self.logger.info("\nLearning Rate: " + str(self.learning_rate))
        self.logger.info("\nNumber of Convolutional Blocks: " + str(cmd_args.num_conv))

        if cmd_args.train:  # Only if training is to be performed
            training_set = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_PATH + '/train/',
                                                      patch_size=self.patch_size,
                                                      samples_per_epoch=self.samples_per_epoch,
                                                      stride_length=self.stride_length, stride_width=self.stride_width,
                                                      stride_depth=self.stride_depth)
            self.train_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=self.num_worker)
            validation_set = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_PATH + '/validate/',
                                                        patch_size=self.patch_size,
                                                        samples_per_epoch=self.samples_per_epoch,
                                                        stride_length=self.stride_length,
                                                        stride_width=self.stride_width,
                                                        stride_depth=self.stride_depth,
                                                        is_train=False)
            self.validate_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=self.num_worker)

    @staticmethod
    def create_tio_sub_ds(vol_path, patch_size, samples_per_epoch, stride_length, stride_width, stride_depth,
                          is_train=True, get_subjects_only=False):

        vols = glob(vol_path + "*.nii") + glob(vol_path + "*.nii.gz")
        subjects = []
        for i in range(len(vols)):
            vol = vols[i]
            filename = os.path.basename(vol).split('.')[0]
            subject = tio.Subject(
                img=tio.ScalarImage(vol),
                subjectname=filename,
            )
            # vol_transforms = tio.ToCanonical(), tio.Resample(tio.ScalarImage(vol))
            # transform = tio.Compose(vol_transforms)
            # subject = transform(subject)
            subjects.append(subject)

        if get_subjects_only:
            return subjects

        if is_train:
            subjects_dataset = tio.SubjectsDataset(subjects)
            sampler = tio.data.UniformSampler(patch_size)
            patches_queue = tio.Queue(
                subjects_dataset,
                max_length=(samples_per_epoch // len(subjects)) * 2,
                samples_per_volume=samples_per_epoch // len(subjects),
                sampler=sampler,
                num_workers=0,
                start_background=True
            )
            return patches_queue
        else:
            overlap = np.subtract(patch_size, (stride_length, stride_width, stride_depth))
            grid_samplers = []
            for i in range(len(subjects)):
                grid_sampler = tio.inference.GridSampler(
                    subjects[i],
                    patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers)

    @staticmethod
    def normaliser(batch):
        for i in range(batch.shape[0]):
            if batch[i].max() > 0.0:
                batch[i] = batch[i] / batch[i].max()
        return batch

    def load(self, checkpoint_path=None, load_best=True):
        if checkpoint_path is None:
            checkpoint_path = self.CHECKPOINT_PATH

        if self.with_apex:
            self.model, self.optimizer, self.scaler = load_model_with_amp(self.model, self.optimizer, checkpoint_path,
                                                                          batch_index="best" if load_best else "last")
        else:
            self.model, self.optimizer = load_model(self.model, self.optimizer, checkpoint_path,
                                                    batch_index="best" if load_best else "last")

    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_similarity_loss = 0
            total_continuity_loss = 0
            total_loss = 0
            batch_index = 0
            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_batch = torch.movedim(local_batch, -1, -3)

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                self.optimizer.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    # Get the classification response map(normalized) and respective class assignments after argmax
                    normalised_res_map = self.model(local_batch)
                    ignore, class_assignments = torch.max(normalised_res_map, 1)

                    # Compute Similarity Loss
                    similarity_loss = self.similarity_loss(normalised_res_map, class_assignments)

                    # Propogate the class probabilities to pixels/voxels
                    local_batch_shape = local_batch.shape
                    class_probablilities = class_assignments.reshape(local_batch_shape).float()

                    # Spatial Continuity comparison
                    cont_width_target = torch.zeros((local_batch_shape[0], local_batch_shape[1],
                                                     local_batch_shape[2]-1, local_batch_shape[3],
                                                     local_batch_shape[4])).float().cuda()
                    cont_length_target = torch.zeros((local_batch_shape[0], local_batch_shape[1],
                                                     local_batch_shape[2], local_batch_shape[3]-1,
                                                     local_batch_shape[4])).float().cuda()
                    cont_height_target = torch.zeros((local_batch_shape[0], local_batch_shape[1],
                                                     local_batch_shape[2], local_batch_shape[3],
                                                     local_batch_shape[4]-1)).float().cuda()
                    cont_width_op = class_probablilities[:, :, 1:, :, :] - class_probablilities[:, :, 0:-1, :, :]
                    cont_length_op = class_probablilities[:, :, :, 1:, :] - class_probablilities[:, :, :, 0:-1, :]
                    cont_height_op = class_probablilities[:, :, :, :, 1:] - class_probablilities[:, :, :, :, 0:-1]
                    continuity_loss_width = self.continuity_loss(cont_width_op, cont_width_target)
                    continuity_loss_length = self.continuity_loss(cont_length_op, cont_length_target)
                    continuity_loss_height = self.continuity_loss(cont_height_op, cont_height_target)
                    avg_continuity_loss = (continuity_loss_width + continuity_loss_length + continuity_loss_height) / 3

                    loss = similarity_loss + (self.cont_loss_coeff * avg_continuity_loss)

                # except Exception as error:
                #     self.logger.exception(error)
                #     sys.exit()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                                 "\n similarity_loss: " + str(similarity_loss) + " continuity_loss: " +
                                 str(avg_continuity_loss) + " total_loss: " + str(loss))

                # Calculating gradients
                if self.with_apex:
                    if type(loss) is list:
                        for i in range(len(loss)):
                            if i + 1 == len(loss):  # final loss
                                self.scaler.scale(loss[i]).backward()
                            else:
                                self.scaler.scale(loss[i]).backward(retain_graph=True)
                        loss = torch.sum(torch.stack(loss))
                    else:
                        self.scaler.scale(loss).backward()

                    if self.clip_grads:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.optimizer.step()

                # if training_batch_index % 50 == 0:  # Save best metric evaluation weights
                #     write_summary(writer=self.writer_training, index=training_batch_index,
                #                   similarity_loss=similarity_loss.detach().item(),
                #                   continuity_loss=avg_continuity_loss.detach().item(),
                #                   total_loss=loss.detach().item())
                training_batch_index += 1

                # Initialising the average loss metrics
                total_similarity_loss += similarity_loss.detach().item()
                total_continuity_loss += avg_continuity_loss.detach().item()
                total_loss += loss.detach().item()

                # To avoid memory errors
                torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_similarity_loss /= (batch_index + 1.0)
            total_continuity_loss /= (batch_index + 1.0)
            total_loss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n similarity_loss: " + str(total_similarity_loss) + " continuity_loss: " +
                             str(total_continuity_loss) + " total_loss: " + str(total_loss))
            write_epoch_summary(writer=self.writer_training, index=epoch,
                                similarity_loss=total_similarity_loss,
                                continuity_loss=total_continuity_loss,
                                total_loss=total_loss)

            save_model(self.CHECKPOINT_PATH, {
                'epoch_type': 'last',
                'epoch': epoch,
                # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved after validate)
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': self.scaler.state_dict()
            })

            torch.cuda.empty_cache()  # to avoid memory errors
            self.validate(training_batch_index, epoch)
            torch.cuda.empty_cache()  # to avoid memory errors

        return self.model

    def validate(self, training_index, epoch):
        """
        Method to validate
        :param training_index: Epoch after which validation is performed(can be anything for test)
        :param epoch: Current training epoch
        :return:
        """
        self.logger.debug('Validating...')
        print("Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))

        total_similarity_loss, total_continuity_loss, total_loss = 0, 0, 0
        no_patches = 0
        self.model.eval()
        data_loader = self.validate_loader
        writer = self.writer_validating
        with torch.no_grad():
            for index, patches_batch in enumerate(tqdm(data_loader)):
                self.logger.info("loading" + str(index))
                no_patches += 1

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_batch = torch.movedim(local_batch, -1, -3)

                try:
                    with autocast(enabled=self.with_apex):
                        # Forward propagation
                        normalised_res_map = self.model(local_batch)
                        ignore, class_assignments = torch.max(normalised_res_map, 1)

                        # Compute Similarity Loss
                        similarity_loss = self.similarity_loss(normalised_res_map, class_assignments)

                        # Propogate the class probabilities to pixels/voxels
                        local_batch_shape = local_batch.shape
                        class_probablilities = class_assignments.reshape(local_batch_shape).float()

                        # Spatial Continuity comparison
                        cont_width_target = torch.zeros((local_batch_shape[0], local_batch_shape[1],
                                                         local_batch_shape[2] - 1, local_batch_shape[3],
                                                         local_batch_shape[4])).float().cuda()
                        cont_length_target = torch.zeros((local_batch_shape[0], local_batch_shape[1],
                                                          local_batch_shape[2], local_batch_shape[3] - 1,
                                                          local_batch_shape[4])).float().cuda()
                        cont_height_target = torch.zeros((local_batch_shape[0], local_batch_shape[1],
                                                          local_batch_shape[2], local_batch_shape[3],
                                                          local_batch_shape[4] - 1)).float().cuda()
                        cont_width_op = class_probablilities[:, :, 1:, :, :] - class_probablilities[:, :, 0:-1, :, :]
                        cont_length_op = class_probablilities[:, :, :, 1:, :] - class_probablilities[:, :, :, 0:-1, :]
                        cont_height_op = class_probablilities[:, :, :, :, 1:] - class_probablilities[:, :, :, :, 0:-1]
                        continuity_loss_width = self.continuity_loss(cont_width_op, cont_width_target)
                        continuity_loss_length = self.continuity_loss(cont_length_op, cont_length_target)
                        continuity_loss_height = self.continuity_loss(cont_height_op, cont_height_target)
                        avg_continuity_loss = (continuity_loss_width + continuity_loss_length +
                                               continuity_loss_height) / 3

                        loss = similarity_loss + (self.cont_loss_coeff * avg_continuity_loss)

                except Exception as error:
                    self.logger.exception(error)

                total_similarity_loss += similarity_loss.detach().item()
                total_continuity_loss += avg_continuity_loss.detach().item()
                total_loss += loss.detach().item()

                # Log validation losses
                self.logger.info("Batch_Index:" + str(index) + " Validation..." +
                                 "\n similarity_loss: " + str(similarity_loss) + " continuity_loss: " +
                                 str(avg_continuity_loss) + " total_loss: " + str(loss))

        # Average the losses
        total_similarity_loss = total_similarity_loss / no_patches
        total_continuity_loss = total_continuity_loss / no_patches
        total_loss = total_loss / no_patches

        process = ' Validating'
        self.logger.info("Epoch:" + str(training_index) + process + "..." +
                         "\n similarity_loss:" + str(total_similarity_loss) +
                         "\n continuity_loss:" + str(total_continuity_loss) +
                         "\n total_loss:" + str(total_loss))

        # write_summary(writer, training_index, similarity_loss=total_similarity_loss,
        #               continuity_loss=total_continuity_loss, total_loss=total_loss)
        write_epoch_summary(writer, epoch, similarity_loss=total_similarity_loss,
                            continuity_loss=total_continuity_loss,
                            total_loss=total_loss)

        if self.LOWEST_LOSS > total_loss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = total_loss
            self.logger.info(
                'Best metric... @ epoch:' + str(training_index) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            save_model(self.CHECKPOINT_PATH, {
                'epoch_type': 'best',
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': self.scaler.state_dict()})

    def test(self, test_logger, test_subjects=None, save_results=True):
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_PATH + '/test/'
            # test_label_path = self.DATASET_PATH + '/test_label/'

            test_subjects = self.create_tio_sub_ds(vol_path=test_folder_path, get_subjects_only=True,
                                                   patch_size=self.patch_size, samples_per_epoch=self.samples_per_epoch,
                                                   stride_depth=self.stride_depth, stride_width=self.stride_width,
                                                   stride_length=self.stride_length)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.OUTPUT_PATH, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            for test_subject in test_subjects:
                if 'label' in test_subject:
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    del test_subject['label']
                else:
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                grid_sampler = tio.inference.GridSampler(
                    test_subject,
                    self.patch_size,
                    overlap,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]

                    local_batch = torch.movedim(local_batch, -1, -3)

                    with autocast(enabled=self.with_apex):
                        normalised_res_map = self.model(local_batch)
                        ignore, class_assignments = torch.max(normalised_res_map, 1)

                        # Propogate the class probabilities to pixels/voxels
                        local_batch_shape = local_batch.shape
                        class_prediction = class_assignments.reshape(local_batch_shape).float()
                        output = torch.movedim(class_prediction, -3, -1).type(local_batch.type())
                        aggregator.add_batch(output, locations)

                predicted = aggregator.get_output_tensor().squeeze().numpy()

                result = predicted.astype(np.float32)

                if label is not None:
                    datum = {"Subject": subjectname}
                    dice3d = dice(result, label)
                    iou3d = iou(result, label)
                    datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3d], "IoU": [iou3d]})
                    df = pd.concat([df, datum], ignore_index=True)

                if save_results:
                    save_nifti(result, os.path.join(result_root, subjectname + "_seg.nii.gz"))

                    # Create Segmentation Mask from the class prediction
                    segmentation_overlay = create_segmentation_mask(result, self.num_classes)
                    save_nifti_rgb(segmentation_overlay, os.path.join(result_root, subjectname + "_seg_color.nii.gz"))
                    # save_tif_rgb(segmentation_overlay, os.path.join(result_root, subjectname + "_colour.tif"))
                    if label is not None:
                        overlay = create_diff_mask_binary(result, label)
                        save_tif_rgb(overlay, os.path.join(result_root, subjectname + "_colour.tif"))

                # test_logger.info("Testing " + subjectname + "..." +
                #                  "\n Dice:" + str(dice3d) +
                #                  "\n JacardIndex:" + str(iou3d))

        # df.to_excel(os.path.join(result_root, "Results_Main.xlsx"))

    def predict(self, image_path, label_path, predict_logger):
        image_name = os.path.basename(image_path).split('.')[0]

        sub_dict = {
            "img": tio.ScalarImage(image_path),
            "subjectname": image_name,
        }

        if bool(label_path):
            sub_dict["label"] = tio.LabelMap(label_path)

        subject = tio.Subject(**sub_dict)

        self.test(predict_logger, test_subjects=[subject], save_results=True)
