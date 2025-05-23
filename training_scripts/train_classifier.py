import os
import yaml
import argparse
from pathlib import Path
from models.classifiers import cls_models
from experiments import exp_modules
from data.dataset_lit import DatasetLit
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/galaxy10_convnext.yaml',
                    )

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)

# For reproducibility
# seed_everything(config['exp_params']['manual_seed'], True)

model = cls_models[config['model_params']['name']](**config['model_params'])
ckpt_path = config['logging_params'].get('ckpt_path', None)
if ckpt_path is not None:
    experiment = exp_modules[config['exp_params']['name']].load_from_checkpoint(checkpoint_path = ckpt_path, cls_model = model, params = config['exp_params'])
    experiment.model.freeze_encoder()
else:
    experiment = exp_modules[config['exp_params']['name']](model,config['exp_params'])
data = DatasetLit(**config['data_params'])
data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                            LearningRateMonitor(),
                            ModelCheckpoint(
                                save_top_k=2,
                                dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                monitor= "valid_auc_epoch",
                                mode='max',
                                save_last= True),
                            ],
                  **config['trainer_params'])

if runner.global_rank == 0:
    Path(f"{tb_logger.log_dir}").mkdir(exist_ok=True, parents=True)
    # Save hparams
    with open(f"{tb_logger.log_dir}/hparams_save.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    print(f"======= Training {config['model_params']['name']} =======")
# runner.fit(experiment, datamodule=data, ckpt_path=config['logging_params'].get('ckpt_path', None))
runner.fit(experiment, datamodule=data)