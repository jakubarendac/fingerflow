import os

from src.fingerflow.matcher.VerifyNet import verify_net_train

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

verify_net_train.train()
