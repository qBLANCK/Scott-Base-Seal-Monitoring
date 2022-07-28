import torch
import os
import copy
import random
import torch.optim as optim
from torch import nn
from dataset.imports import load_dataset
from detection.retina import model as retina
import checkpoint
from tools import struct, logger, Struct
from tools.logger import EpochLogger
import trainer
import evaluate
import math
import arguments
import pprint

import sys
from gpu_profile import gpu_profile

pp = pprint.PrettyPrinter(indent=2)


def log_anneal(range, t):
    begin, end = range
    return math.exp(math.log(begin) * (1 - t) + math.log(end) * t)


def cosine_anneal(range, t):
    begin, end = range
    return end + 0.5 * (begin - end) * (1 + math.cos(t * math.pi))


def schedule_lr(t, epoch, args):
    lr_min = args.lr * args.lr_min

    if args.lr_decay == "log":
        return log_anneal((args.lr, lr_min), t)
    elif args.lr_decay == "cosine":
        return cosine_anneal((args.lr, lr_min), t)
    elif args.lr_decay == "step":
        n = math.floor(epoch / args.lr_schedule)
        return max(lr_min, args.lr * math.pow(args.lr_step, -n))
    else:
        assert False, "unknown lr decay method: " + args.lr_decay


def set_bn_momentum(model, mom):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = mom


def get_nms_params(args):
    return struct(
        nms=args.nms_threshold,
        threshold=args.class_threshold,
        detections=args.max_detections)


class Trainer():
    def __init__(self) -> None:
        args = arguments.get_arguments()
        pp.pprint(args._to_dicts())
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        config, dataset = load_dataset(args)

        # Initialise
        data_root = config.root
        log_root = args.log_dir or data_root

        model_args = struct(
            dataset=struct(
                classes=dataset.classes,
                input_channels=3),
            version=2,
            model_params=args.model_params
        )

        run = 0

        debug = struct(
            predictions=args.debug_predictions or args.debug_all,
            boxes=args.debug_boxes or args.debug_all
        )

        output_path, log = logger.make_experiment(
            log_root, args.run_name, load=not args.no_load, dry_run=args.dry_run)
        model_path = os.path.join(output_path, "model.pth")

        model, encoder = retina.create(args.model_params, model_args.dataset)

        set_bn_momentum(model, args.bn_momentum)

        best, current, _ = checkpoint.load_checkpoint(
            model_path, model, model_args, args)
        model, epoch = current.model, current.epoch + 1

        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

        device = torch.cuda.current_device()
        tests = args.tests.split(",")

        # Allocate more GPU memory
        # fraction = 3 / 4
        # torch.cuda.set_per_process_memory_fraction(fraction, device)

        self.args = args
        self.epoch = epoch
        self.optimizer = optimizer
        self.dataset = dataset
        self.log = log
        self.model = model.to(device)
        self.device = device
        self.encoder = encoder.to(device)
        self.debug = debug
        self.best = best
        self.tests = tests
        self.run = run
        self.model_path = model_path
        self.model_args = model_args

    def wrap_up(self, msg):
        print(f"Learning completed!\nReason: {msg}")
        exit()

    def adjust_learning_rate(self, n, total):
        lr = schedule_lr(n/total, self.epoch, self.args)
        for param_group in self.optimizer.param_groups:
            modified = lr * \
                param_group['modifier'] if 'modifier' in param_group else lr
            param_group['lr'] = modified

    def test_images(self, images, split=False, hook=None):
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
        return trainer.test(self.dataset.test_on(images, self.args, self.encoder), eval_test, hook=hook)

    def run_testing(self, name, images, split=False, hook=None, thresholds=None):
        if len(images) > 0:
            print("{} {}:".format(name, self.epoch))
            results = self.test_images(images, split=split, hook=hook)

            return evaluate.summarize_test(name, results, self.dataset.classes, self.epoch,
                                           log=EpochLogger(self.log, self.epoch), thresholds=thresholds)

        return 0, None

    def training_cycle(self):
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
        train_stats = trainer.train(self.dataset.sample_train_on(train_images, self.args, self.encoder),
                                    evaluate.eval_train(self.model.train(), self.encoder, self.debug,
                                                        device=self.device), self.optimizer, hook=self.adjust_learning_rate)
        gpu_profile(frame=sys._getframe(), event='line', arg=None)
        torch.cuda.empty_cache()
        gpu_profile(frame=sys._getframe(), event='line', arg=None)
        evaluate.summarize_train("train", train_stats,
                                 self.dataset.classes, self.epoch, log=log)
        gpu_profile(frame=sys._getframe(), event='line', arg=None)
        score, thresholds = self.run_testing(
            'validate', self.dataset.validate_images, split=self.args.eval_split == True)

        is_best = score >= self.best.score
        if is_best:
            self.best = struct(model=copy.deepcopy(
                self.model), score=score, thresholds=thresholds, epoch=self.epoch)

        current = struct(state=self.model.state_dict(),
                         epoch=self.epoch, thresholds=thresholds, score=score)
        best = struct(state=self.best.model.state_dict(
        ), epoch=self.best.epoch, thresholds=self.best.thresholds, score=self.best.score)

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
    trainer = Trainer()
    try:
        while(True):
            trainer.training_cycle()
    except RuntimeError as error:
        print(torch.cuda.memory_summary())
        raise RuntimeError(error)


if __name__ == '__main__':
    run_main()
