import copy
import math
import os
import pprint
import random

import torch
import torch.optim as optim
from torch import nn

import Models.Seals.arguments as arguments
import Models.Seals.checkpoint as checkpoint
import Models.Seals.evaluate as evaluate
import Models.Seals.trainer as trainer
from Models.Seals.dataset.imports import load_dataset
from Models.Seals.detection.retina import model as retina
from libs.tools import struct, logger, Struct
from libs.tools.logger import EpochLogger

pp = pprint.PrettyPrinter(indent=2)


def schedule_lr(t, args):
    """Return the learning rate for the current time. Using Time-based decay to decay lr during training.
    Log decay with minimum lr."""
    lr_min = args.lr * args.lr_min
    return math.exp(math.log(args.lr) * (1 - t) + math.log(lr_min) * t)


def set_bn_momentum(model, mom):
    """Set the momentum for all batch normalisation modules in the model."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = mom


def get_nms_params(args):
    """Return the parameters for non-maximum suppression."""
    return struct(
        nms=args.nms_threshold,
        threshold=args.class_threshold,
        detections=args.max_detections)


class Trainer():
    """Train the model."""

    def __init__(self) -> None:
        """Initialize the trainer."""
        args = arguments.get_arguments()
        pp.pprint(args._to_dicts())
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        config, self.dataset = load_dataset(args)

        # Initialise
        self.model_args = struct(
            dataset=struct(
                classes=self.dataset.classes,
                input_channels=3),
            version=2,
            model_params=args.model_params
        )
        self.debug = struct(
            predictions=args.debug_predictions or args.debug_all,
            boxes=args.debug_boxes or args.debug_all
        )
        log_root = args.log_dir or config.root
        output_path, self.log = logger.make_experiment(
            log_root, args.run_name, load=not args.no_load, dry_run=args.dry_run)
        self.model_path = os.path.join(output_path, "model.pth")
        model, encoder = retina.create(
            args.model_params, self.model_args.dataset)
        set_bn_momentum(model, args.bn_momentum)
        self.best, current, _ = checkpoint.load_checkpoint(
            self.model_path, model, self.model_args, args)
        model, self.epoch = current.model, current.epoch + 1
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        self.device = torch.cuda.current_device()
        self.tests = args.tests.split(",")
        self.args = args
        self.model = model.to(self.device)
        self.encoder = encoder.to(self.device)
        self.run = 0

    def wrap_up(self, msg):
        """Wrap up the training."""
        print(f"Learning completed!\nReason: {msg}")
        exit()

    def adjust_learning_rate(self, n, total):
        """Adjust the learning rate."""
        lr = schedule_lr(n / total, self.args)
        for param_group in self.optimizer.param_groups:
            modified = lr * \
                param_group['modifier'] if 'modifier' in param_group else lr
            param_group['lr'] = modified

    def test_images(self, images, split=False, hook=None):
        """Test the model on a set of images."""
        eval_params = struct(
            overlap=self.args.overlap,
            split=split,
            image_size=(self.args.image_size, self.args.image_size),
            batch_size=self.args.batch_size,
            nms_params=get_nms_params(self.args),
            device=self.device,
            debug=self.debug
        )

        eval_test = evaluate.eval_test(
            self.model.eval(), self.encoder, eval_params)
        return trainer.test(self.dataset.test_on(
            images, self.args, self.encoder), eval_test, hook=hook)

    def run_testing(self, name, images, split=False,
                    hook=None, thresholds=None):
        """Run the testing."""
        if len(images) > 0:
            print("{} {}:".format(name, self.epoch))
            results = self.test_images(images, split=split, hook=hook)

            return evaluate.summarize_test(
                name, results, self.dataset.classes, self.epoch, log=EpochLogger(
                    self.log, self.epoch), thresholds=thresholds)

        return 0, None

    def training_cycle(self):
        """Run a training cycle."""
        if len(self.dataset.train_images) == 0:
            raise Exception("Either no environment or training dataset")

        if self.args.max_epochs is not None and self.epoch > self.args.max_epochs:
            self.wrap_up(f"Max epochs ({self.args.max_epochs}) reached.")

        log = EpochLogger(self.log, self.epoch)

        log.scalars("dataset", Struct(self.dataset.count_categories()))

        train_images = self.dataset.train_images
        if self.args.incremental is True:
            t = self.epoch / self.args.max_epochs
            n = max(1, min(int(t * len(train_images)), len(train_images)))
            train_images = train_images[:n]

        print("Training {} on {} images:".format(
            self.epoch, len(train_images)))
        train_stats = trainer.train(
            self.dataset.sample_train_on(
                train_images,
                self.args,
                self.encoder),
            evaluate.eval_train(
                self.model.train(),
                self.encoder,
                self.debug,
                device=self.device),
            self.optimizer,
            hook=self.adjust_learning_rate)
        torch.cuda.empty_cache()
        evaluate.summarize_train("train", train_stats,
                                 self.dataset.classes, self.epoch, log=log)
        score, thresholds = self.run_testing(
            'validate', self.dataset.validate_images, split=self.args.eval_split)

        is_best = score >= self.best.score
        if is_best:
            self.best = struct(
                model=copy.deepcopy(
                    self.model),
                score=score,
                thresholds=thresholds,
                epoch=self.epoch)

        current = struct(state=self.model.state_dict(),
                         epoch=self.epoch, thresholds=thresholds, score=score)
        best = struct(
            state=self.best.model.state_dict(),
            epoch=self.best.epoch,
            thresholds=self.best.thresholds,
            score=self.best.score)

        for test_name in self.tests:
            self.run_testing(test_name, self.dataset.get_images(
                test_name), thresholds=self.best.thresholds)

        save_checkpoint = struct(
            current=current, best=best, args=self.model_args, run=self.run)
        torch.save(save_checkpoint, self.model_path)

        self.epoch = self.epoch + 1

        if self.best.epoch < self.epoch - self.args.validation_pause:
            self.wrap_up(
                f"Validation not improved after {self.args.validation_pause} epochs.")
        else:
            print(f"Best epoch: {self.best.epoch}")

        log.flush()


def run_main():
    """Run the trainer."""
    trainer = Trainer()
    try:
        while (True):
            trainer.training_cycle()
    except RuntimeError as error:
        print(torch.cuda.memory_summary())
        raise RuntimeError(error)


if __name__ == '__main__':
    run_main()
