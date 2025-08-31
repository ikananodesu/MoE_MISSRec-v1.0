import os
import argparse
from logging import getLogger
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from recbole.utils import init_seed, init_logger

from config import Config
from missrec import MISSRec
from data.dataset import PretrainMISSRecDataset
from data.dataloader import CustomizedTrainDataLoader
from trainer import DDPMISSRecPretrainTrainer

def pretrain(rank, world_size, dataset, **kwargs):
    """
    分布式预训练主函数。
    每个进程（GPU）都会调用本函数，各自负责一部分数据和模型训练。

    参数:
        rank: 当前进程编号（GPU编号）
        world_size: 总进程（GPU）数
        dataset: 数据集名称
        kwargs: 其它配置参数
    """
    props = ['props/MISSRec.yaml', 'props/pretrain.yaml']
    if rank == 0:
        print('DDP Pre-training on:', dataset)
        print(props)
    kwargs.update({'ddp': True, 'rank': rank, 'world_size': world_size})
    config = Config(model=MISSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    if config['rank'] not in [-1, 0]:
        config['state'] = 'warning'
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    dataset = PretrainMISSRecDataset(config)
    logger.info(dataset)
    pretrain_dataset = dataset.build()[0]  # 只取训练部分
    pretrain_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)
    model = MISSRec(config, pretrain_data.dataset)
    logger.info(model)
    trainer = DDPMISSRecPretrainTrainer(config, model) #需要重构
    trainer.pretrain(pretrain_data, show_progress=(rank == 0))
    dist.destroy_process_group()
    return config['model'], config['dataset']

if __name__ == '__main__':
    """
    程序主入口。用于解析命令行参数，初始化多卡环境，并启动分布式训练。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='FHCKM_mm_full', help='dataset name')
    parser.add_argument('-p', type=str, default='12355', help='port for ddp')
    args, unparsed = parser.parse_known_args()

    # 1. 检查GPU数，必须>=2
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}."
    world_size = n_gpus

    # 2. 配置DDP主机与端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.p

    # 3. 启动多进程，每GPU一个进程
    mp.spawn(
        pretrain,  # 每个进程执行pretrain
        args=(world_size, args.d,),  # 传递world_size, dataset
        nprocs=world_size,  # 启动的进程数
        join=True
    )