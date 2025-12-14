from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'DiffusionMSE_CFG'
_C.MODEL.ENCODER = 'basicGEBD'
_C.MODEL.DECODER = 'transformer'
_C.MODEL.CLASS_DIM = 1
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.BACKBONE.FREEZE = False
_C.MODEL.SYNC_BN = True
_C.MODEL.DIMENSION = 512
_C.MODEL.FEED_FORWARD_DIM = 2048
_C.MODEL.NUM_LAYERS = 6
_C.MODEL.NUM_HEADS = 8
_C.MODEL.WINDOW_SIZE = 17 
_C.MODEL.FPN_START_IDX = 0 
_C.MODEL.HEAD_CHOICE = 1 
_C.MODEL.NUM_BLOCKS = 3
_C.MODEL.ENCODER_OUT = False
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ('GEBD_train',)
_C.DATASETS.TEST = ('GEBD_val',)

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.RESOLUTION = 224
_C.INPUT.ARGUMENT = True
_C.INPUT.ANNOTATORS = 2
_C.INPUT.ANNOTATOR_SELECT = 'best'
_C.INPUT.FRAME_PER_SIDE = 5
_C.INPUT.DYNAMIC_DOWNSAMPLE = False
_C.INPUT.GAUSSIAN_TARGET = True
_C.INPUT.GAUS_SIGMA = 1.
_C.INPUT.ONLY_TARGET_GAUS = False
_C.INPUT.DOWNSAMPLE = 3
_C.INPUT.END_TO_END = True  # input whole video
_C.INPUT.SEQUENCE_LENGTH = 50  # input whole video
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 30
_C.SOLVER.WARMUP = True
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.SCHEDULER = 'cosine'
_C.SOLVER.MILESTONES = [2, 3]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.AMPE = True  # automatic mixed precision training
_C.SOLVER.LR = 1e-2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.CLIP_GRAD = 0.0
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.OPTIMIZER = 'SGD'
# ---------------------------------------------------------------------------- #
# Diffusion
# ---------------------------------------------------------------------------- #
_C.DIFFUSION = CN()
_C.DIFFUSION.DETERMINISTIC = False
_C.DIFFUSION.CFG_PROB = 0.1
_C.DIFFUSION.CFG_SCALE = 0.0
_C.DIFFUSION.BETA_SCHEDULE = 'cosine'
_C.DIFFUSION.TIMESTEPS = 1000
_C.DIFFUSION.SAMPLING_TIMESTEPS = 16
_C.DIFFUSION.VALIDATION_TIMESTEPS = [0]
_C.DIFFUSION.DDIM_SAMPLING_ETA = 0.0
_C.DIFFUSION.SNR_SCALE = 0.5
# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.THRESHOLD = 0.5
_C.TEST.PRED_FILE = ''  # precomputed predictions
_C.TEST.SMOKE_TEST = False
_C.TEST.PROTOCOL = 'max'
# ---------------------------------------------------------------------------- #
# OTHERS
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = 'output'
