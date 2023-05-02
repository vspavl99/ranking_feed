import math

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import MultiVAE
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import torch

from src.features.embeding_as_layer import load_embedding, resize_embedding

MODEL_NAME = 'MultiVAE'


def train_MultiVAE(config):

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = MultiVAE(config, train_data.dataset)

    if config['use_embedding']:
        logger.info('Loading embedding for initializing last layer')
        embeddings = load_embedding(train_data.dataset)

        in_dim, out_dim = model.decoder[-1].in_features, model.decoder[-1].out_features
        assert len(embeddings) == out_dim
        logger.info(f'Embedding will be projected to {in_dim}')
        print(embeddings.shape)
        # projected_embeddings = resize_embedding(embeddings, in_dim)
        projected_embeddings = embeddings

        print(embeddings.shape)
        model.decoder[-1].weight.data = (torch.Tensor(projected_embeddings) - 0) / (1 * math.sqrt(2 / float(in_dim + out_dim)))
    model = model.to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)
    #
    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=config["show_progress"])

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)


if __name__ == '__main__':

    cfg = Config(
        model=MODEL_NAME,
        dataset='mind_small',
        config_file_list=[f'config/experiments_external_data/{MODEL_NAME}_model.yaml'],
        config_dict={
            'log_wandb': True,
            'train_neg_sample_args': None,
            'use_embedding': False
        }
    )

    train_MultiVAE(cfg)
