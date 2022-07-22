from tools.parameters import param, parse_args, choice, parse_choice, group, add_arguments
from tools import struct
import argparse

from detection.retina import model as retina

train_parameters = struct (
    optimizer = group('optimizer settings',
        lr              = param(0.001,    help='learning rate'),

        lr_decay        = param('log', help='type of LR decay to use (log|cosine|step)'),
       
        lr_schedule     = param(30, type='int', help='epochs before dropping LR for step decay'),
        lr_step         = param(10, type='float', help='step factor to drop LR'),

        lr_min          = param(0.1,    help='minimum learning rate to decay to (factor of initial lr)'),
        
        momentum        = param(0.5,    help='SGD momentum'),
        weight_decay    = param(1e-4, help='weight decay rate')
    ),

    seed            = param(1,      help='random seed'),
    batch_size      = param(8,     help='input batch size for training'),

    tests = param('', help='comma separated list of test sets to use'),
    
    epoch_size          = param(1024, help='epoch size for training'),
    validation_pause    = param(16,    type='int', help='automatically pause training if validation does not improve after epochs'),

    incremental       = param(False, help='simulate incremental adding to the dataset during training'),
    max_epochs      = param(None, type='int', help='maximum number of epochs to train'),

    eval_split      = param(False, help='evaluate images split into crops of image_size'),
    overlap         = param(200, type='int', help='margin of overlap when splitting images for evaluation'),

    num_workers     = param(4,      help='number of workers used to process dataset'),

    bn_momentum    = param(0.9, "momentum for batch normalisation modules"),

    no_load         = param(False,   help="don't attempt to load previously saved model"),
    dry_run         = param(False,   help="run for testing only (don't store results or log progress)"),

    restore_best    = param(False,   help="restore weights from best validation model"),

    log_dir         = param(None, type='str', help="output directory for logging"),
    run_name        = param('training', help='name for training run')
)


detection_parameters = struct (
    image = group('image',
        gamma           = param(0.15,  help='variation in gamma when training'),
        channel_gamma   = param(0.0,  help='variation per channel gamma when training'),

        brightness      = param(0.05,  help='variation in brightness (additive) when training'),
        contrast        = param(0.05,  help='variation in contrast (multiplicative) when training'),

        hue             = param(0.05,  help='variation in brightness (additive) when training'),
        saturation      = param(0.00,  help='variation in contrast (multiplicative) when training'),        

        train_size      = param(600,   help='size of patches to train on'),

        border_bias     = param(0.1,    help = "bias random crop to select border more often (proportion of size)"),
        augment = param("crop", help = 'image augmentation method (crop | full | ssd)'),

        scale  = param(1.0,     help='base scale of train_size'),
        resize = param(None, type='float', help='resize short side of images to this dimension'),

        max_scale = param(4./3, help='maximum scale multiplier for cropped patches'),
        min_scale = param(None, type='float', help = 'minimum scale multiplier for cropped patches (default 1/max_scale)'),

        max_aspect = param(1.1, help = 'maximum aspect ratio on crops'),

        transposes  = param(False, help='enable image transposes in training'),
        flips          = param(True, help='enable horizontal image flips in training'),
        vertical_flips = param(False, help='enable vertical image flips in training'),
        image_samples   = param(1,      help='number of training samples to extract from each loaded image')
    ),


    nms = group('nms',
        nms_threshold    = param (0.5, help = "overlap threshold (iou) used in nms to filter duplicates"),
        class_threshold  = param (0.05, help = 'hard threshold used to filter negative boxes'),
        max_detections    = param (100,  help = 'maximum number of detections (for efficiency) in testing')
    ),

    select_instance = param(0.5, help = 'probability of cropping around an object instance as opposed to a random patch'),
    min_visible     = param (0.4, help = 'minimum proportion of area for an overlapped box to be included')
)


debug_parameters =  struct (
    debug = group('debug',
        debug_predictions = param (False, "distribution of predictions by class"),
        debug_boxes       = param (False, "debug anchor box matches"),
        debug_all         = param (True, "enable all debug options")
    )
)


input_choices = struct(
    json = struct(
        path          = param(type="str", help = "path to exported json annotation file", required=True)),
    coco = struct(
        path          = param(type="str", help = "path to exported json annotation file", required=True),
        image_root    = param(type="str", help = "path to root of training images", required=True),
        split_ratio   = param(type="str", help = "slash separated list of dataset ratio (train/validate/test) e.g. 70/15/15", default="70/15/15"))
)

input_parameters = struct(
        input        = choice(default="json", options=input_choices, help='input method'),
    )


parameters = detection_parameters._merge(train_parameters)._merge(input_parameters)._merge(debug_parameters)._merge(retina.parameters)


def get_arguments():
    args = parse_args(parameters, "trainer", "object detection parameters")
    args.input = parse_choice("input", parameters.input, args.input)
    
    parser = argparse.ArgumentParser()
    add_arguments(parser, retina.parameters)
    _args = parser.parse_known_args()
    args.model_params = struct(**_args[0].__dict__)

    return args
