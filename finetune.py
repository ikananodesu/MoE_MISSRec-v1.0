import argparse
from logging import getLogger, FileHandler
import torch
from collections import OrderedDict
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color

from missrec import MISSRec
from data.dataset import MISSRecDataset
from trainer import MISSRecTrainer

def get_logger_filename(logger):
    file_handler = next((handler for handler in logger.handlers if isinstance(handler, FileHandler)), None)
    if file_handler:
        filename = file_handler.baseFilename
        print(f"The log file name is {filename}")
    else:
        raise Exception("No file handler found in logger")
    return filename

def finetune(
    dataset, 
    pretrained_file, 
    centroids_file,
    props='props/MISSRec.yaml,props/finetune.yaml', 
    mode='transductive', 
    fix_enc=True, 
    log_prefix="", 
    **kwargs
):
    # 配置初始化
    props = props.split(',')
    config = Config(model=MISSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    config['log_prefix'] = log_prefix

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    if config['train_stage'] != mode + '_ft':
        logger.info(f"Change train stage from '{config['train_stage']}' to '{mode + '_ft'}'")
        config['train_stage'] = mode + '_ft'
    logger.info(config)

    # 数据集准备
    dataset_obj = MISSRecDataset(config)
    logger.info(dataset_obj)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    # 模型初始化
    model = MISSRec(config, train_data.dataset).to(config['device'])

    # 加载预训练参数
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file, map_location=config['device'])
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        # moe_adaptor命名适配
        new_state_dict = OrderedDict()
        for name, val in checkpoint['state_dict'].items():
            if name.startswith('moe_adaptor'):
                new_state_dict[name.replace('moe_adaptor', 'text_moe_adaptor')] = val
            else:
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict, strict=False)
        # 参数冻结
        if fix_enc:
            logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_model.parameters():
                _.requires_grad = False
        if hasattr(model, 'mm_fusion'):
            for _ in model.mm_fusion.parameters():
                _.requires_grad = False

    logger.info(model)

    # 统计可训练参数
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    logger.log(level=20, msg=f'Trainable parameters: {trainable_params}')

    # 加载centroids
    centroids = torch.load(centroids_file, map_location=config['device'])
    logger.info(f"Loaded centroids shape: {centroids.shape}")

    # Trainer初始化
    trainer = MISSRecTrainer(config, model)
    trainer.centroids = centroids

    # 训练
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # 测试评估
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    logger_Filename = get_logger_filename(logger)
    logger.info(f"Write log to {logger_Filename}")

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Scientific_mm_full', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-centroids', type=str, default='centroids.pth', help='centroids path')
    parser.add_argument('-f', type=bool, default=True)
    parser.add_argument('-props', type=str, default='props/MISSRec.yaml,props/finetune.yaml')
    parser.add_argument('-mode', type=str, default='transductive') # transductive (w/ ID) / inductive (w/o ID)
    parser.add_argument('-note', type=str, default='')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(
        args.d,
        props=args.props,
        mode=args.mode,
        pretrained_file=args.p,
        centroids_file=args.centroids,
        fix_enc=args.f,
        log_prefix=args.note
    )