import torch.nn
from model.ours import Ours
import time
from model.seed import *
from model.dataset import *
import gc
import argparse
from tqdm import tqdm

from fvcore.nn import FlopCountAnalysis

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser(description='Training for Steering Wheel Datasets.')

    # 随机数种子
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    # 数据集位置
    if os.path.exists(os.path.join('data')):
        parser.add_argument('--data_path', default=os.path.join('E:\\', 'Project', 'SW', 'data'), type=str,
                            help='Root directory of the dataset.')
    else:
        parser.add_argument('--data_path', default='/root/autodl-fs/SW', type=str,
                            help='Root directory of the dataset.')

    # 模型
    parser.add_argument('--model_name', type=str, default=None, help='Model name.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of the model.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of MLP layers in the model.')
    parser.add_argument('--input_dim_edge', type=int, default=4, help='Position feature size.')
    parser.add_argument('--output_dim', type=int, default=1, help='Output feature size.')
    parser.add_argument('--input_dim_node', type=int, default=4, help='Node feature size.')

    parser.add_argument('--num_clusters', type=int, default=64, help='Number of clusters.')
    parser.add_argument('--message_passing_steps', type=int, default=5, help='Number of message passing steps.')
    parser.add_argument('--transformer_block', type=int, default=5, help='Message dimension.')

    # 其他参数
    parser.add_argument('--apply_noise', type=bool, default=True, help='Whether to apply noise.')
    parser.add_argument('--noise', type=float, default=2e-2, help='Noise level.')

    # 批处理
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')

    # 训练
    parser.add_argument('--num_epochs', type=int, default=0, help='Number of epochs to train. If epoch==0, test')

    # 学习率
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--decayRate', type=float, default=0.99999, help='Decay rate for learning rate.')

    # 评估和保存
    parser.add_argument('--save_epoch', type=int, default=10, help='Number of steps between saving the model.')
    parser.add_argument('--ckpt_root', type=str, default='ckpt-sw', help='Directory to save checkpoints.')
    parser.add_argument('--result_root', type=str, default='result-sw', help='Save Results.')

    return parser.parse_args()


def get_loss(output, output_hat, normalized_output, normalized_output_hat, mask, criterion):
    '''
    :param output: [B, N, 1]
    :param output_hat: [B, N, 1]
    :param normalized_output: [B, N, 1]
    :param normalized_output_hat: [B, N, 1]
    :param mask: [B, N, 1]
    :param criterion: 损失函数
    :param config: 配置参数
    :return: 损失字典
    '''
    # 计算 RMSE
    rmse = torch.sqrt(((output * mask - output_hat * mask) ** 2).mean() * mask.numel() / mask.sum())

    # 计算损失
    normalized_loss = criterion(normalized_output * mask, normalized_output_hat * mask) * mask.numel() / mask.sum()

    # 返回损失字典
    losses = {
        'RMSE': rmse,
        'loss': normalized_loss
    }
    return losses


def monitor_memory(epoch=None):
    # Monitor memory usage
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_reserved = torch.cuda.memory_reserved(device)
    if epoch is not None:
        print(f"Epoch {epoch + 1} Memory Allocated: {memory_allocated / (1024 ** 2):.2f} MB")
        print(f"Epoch {epoch + 1} Memory Reserved: {memory_reserved / (1024 ** 2):.2f} MB")
    else:
        print(f"Memory Allocated: {memory_allocated / (1024 ** 2):.2f} MB")
        print(f"Memory Reserved: {memory_reserved / (1024 ** 2):.2f} MB")
    return


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def train(model, data_loader, criterion, optimizer, scheduler, device, num_epochs, val_loader, test_loader):
    os.makedirs(config.ckpt_root, exist_ok=True)
    best_loss = float('inf')
    best_epoch = -1

    set_seed(config.seed)

    # # 如果有best_model,就预加载，否则从头开始训练
    # if os.path.exists(f'{config.ckpt_root}/best_model.pth'):
    #     model.load_state_dict(torch.load(f'{config.ckpt_root}/best_model.pth'))
    #     model = model.to(device)
    #     best_epoch = 0
    #     best_loss, _, _ = validate(model, val_loader, criterion, device)

    train_rmse_values = []
    valid_rmse_values = []

    for i, (batch, _, _) in enumerate(data_loader):
        pos = batch['pos'].to(device)  # [B, N, 2]
        node = batch['node'].to(device)  # [B, N, 4]
        connections = batch['connections'].to(device)  # [B, E, 2]
        output = batch['output'].to(device)  # [B, N, 1]
        model.accumulate(node, pos, connections, output)

    for epoch in range(num_epochs):
        model.train()

        start_time = time.time()
        total_loss = 0
        total_var_loss = 0
        total_RMSE = 0

        for i, (batch, _, _) in enumerate(data_loader):
            set_seed(config.seed)
            pos = batch['pos'].to(device)  # [B, N, 2]
            node = batch['node'].to(device)  # [B, N, 4]
            connections = batch['connections'].to(device)  # [B, E, 2]
            mask = batch['mask'].to(device)  # [B, N]
            output = batch['output'].to(device)  # [B, N, 1]

            # output_hat:预测的output
            # normalized_output: 归一化后的output
            # normalized_output_hat: 归一化后的output_hat
            output_hat, normalized_output, normalized_output_hat = model(pos,
                                                                         node,
                                                                         connections,
                                                                         output,
                                                                         mask=mask,
                                                                         noise=True,
                                                                         mode='train')

            # Compute the loss
            mask = mask.unsqueeze(-1)
            costs = get_loss(output, output_hat, normalized_output, normalized_output_hat, mask, criterion)
            loss, RMSE = costs['loss'], costs['RMSE']

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_RMSE += RMSE

            clear_memory()

        # 保存模型
        if (epoch + 1) % config.save_epoch == 0:
            torch.save(model.state_dict(), f'{config.ckpt_root}/{epoch + 1}.pth')

        clear_memory()

        if scheduler.get_last_lr()[0] > 1e-6 and epoch > 1:
            scheduler.step()

        train_loss = total_loss / len(data_loader)
        total_RMSE = total_RMSE / len(data_loader)

        val_loss, val_RMSE = validate(model, val_loader, criterion, device, epoch, config)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Train RMSE: {total_RMSE:.6f}, "
            f"Validation Loss: {val_loss:.6f}, "
            f"Validation RMSE: {val_RMSE:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )

        train_rmse_values.append(total_RMSE.detach())
        valid_rmse_values.append(val_RMSE.detach())
        np.savez(os.path.join(config.ckpt_root, 'rmse_values.npz'),
                 train_rmse=torch.tensor(train_rmse_values).cpu().numpy(),
                 valid_rmse=torch.tensor(valid_rmse_values).cpu().numpy())

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_RMSE = val_RMSE
            print(
                f"Best Epoch: {best_epoch}, "
                f"Best Loss: {best_loss:.6f}, "
                f"Best RMSE: {best_RMSE:.6f}")
            torch.save(model.state_dict(), f'{config.ckpt_root}/best_model.pth')

            _, _ = test(model, test_loader, criterion, device, epoch, config)

        clear_memory()

    return best_epoch


def validate(model, data_loader, criterion, device, epoch, config):
    model.eval()
    total_loss = 0
    total_RMSE = 0

    with torch.no_grad():
        for i, (batch, _, _) in enumerate(data_loader):
            set_seed(config.seed)
            pos = batch['pos'].to(device)  # [B, N, 2]
            node = batch['node'].to(device)  # [B, N, 4]
            connections = batch['connections'].to(device)  # [B, E, 2]
            mask = batch['mask'].to(device)  # [B, N]
            output = batch['output'].to(device)  # [B, N, 1]

            # 模型前向传播
            output_hat, normalized_output, normalized_output_hat = model(pos,
                                                                         node,
                                                                         connections,
                                                                         output,
                                                                         mask=mask,
                                                                         noise=False,
                                                                         mode='validate')

            # 计算损失
            mask = mask.unsqueeze(-1)
            costs = get_loss(output, output_hat, normalized_output, normalized_output_hat, mask, criterion)
            loss, RMSE = costs['loss'], costs['RMSE']

            total_loss += loss.item()
            total_RMSE += RMSE

            # 清理内存
            clear_memory()

    val_loss = total_loss / len(data_loader)
    val_RMSE = total_RMSE / len(data_loader)

    return val_loss, val_RMSE


def test(model, data_loader, criterion, device, epoch, config):
    model.eval()
    begin_time = time.time()

    total_loss = 0
    total_RMSE = 0

    if not os.path.exists(config.result_root):
        os.mkdir(config.result_root)

    with torch.no_grad():
        for i, (batch, idx, path) in enumerate(data_loader):
            set_seed(config.seed)
            pos = batch['pos'].to(device)  # [B, N, 2]
            node = batch['node'].to(device)  # [B, N, 4]
            connections = batch['connections'].to(device)  # [B, E, 2]
            mask = batch['mask'].to(device)  # [B, N]
            output = batch['output'].to(device)  # [B, N, 1]

            # 模型前向传播
            output_hat, normalized_output, normalized_output_hat = model(pos,
                                                                         node,
                                                                         connections,
                                                                         output,
                                                                         mask=mask,
                                                                         noise=False,
                                                                         mode='test')

            # 计算损失
            mask = mask.unsqueeze(-1)
            costs = get_loss(output, output_hat, normalized_output, normalized_output_hat, mask, criterion)
            loss, RMSE = costs['loss'], costs['RMSE']

            total_loss += loss.item()
            total_RMSE += RMSE

            # 保存预测结果
            for b in range(output_hat.shape[0]):
                valid_nodes = mask[b].bool()
                np.savez(f'{config.result_root}/{path[b]}',
                         prediction=output_hat[b, valid_nodes].detach().cpu().numpy(),
                         groundtruth=output[b, valid_nodes].cpu().numpy())

            # 清理内存
            clear_memory()

    test_loss = total_loss / len(data_loader)
    test_RMSE = total_RMSE / len(data_loader)

    end_time = time.time()

    print(
        f"Test Time: {end_time - begin_time:.2f}s, "
        f"Test Loss: {test_loss:.6f}, "
        f"Test RMSE: {test_RMSE:.6f}"
    )

    return test_loss, test_RMSE


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    print(f'Total number of parameters: {total_params_in_millions:.2f}M')
    return


if __name__ == '__main__':
    config = get_arguments()

    # 设置随机种子
    set_seed(config.seed)

    train_dataset = SWDataset(data_path=config.data_path, mode='train')
    valid_dataset = SWDataset(data_path=config.data_path, mode='valid')
    test_dataset = SWDataset(data_path=config.data_path, mode='test')

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             collate_fn=collate_fn)

    model = Ours(config)
    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.decayRate)

    criterion = torch.nn.MSELoss()

    # 将模型和数据迁移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    for i, batch in enumerate(train_loader):
        if i == 0:  # 第二个 batch 的索引是 1
            dummy_batch = batch[0]
            break

    dummy_pos = dummy_batch['pos'].to(device)  # [B, N, 2]
    dummy_node = dummy_batch['node'].to(device)  # [B, N, input_dim_node]
    dummy_connections = dummy_batch['connections'].to(device)  # [B, E, 2]
    dummy_output = dummy_batch['output'].to(device)  # [B, N, 1]
    dummy_mask = dummy_batch['mask'].to(device)  # [B, N]


    # 自定义包装类
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super(WrappedModel, self).__init__()
            self.model = model

        def forward(self, pos, node, connections, output, mask):
            return self.model(pos, node, connections, output, mask=mask, noise=False, mode='test')[0]


    # 使用包装类
    wrapped_model = WrappedModel(model)

    # 执行 FLOPs 分析
    flops = FlopCountAnalysis(wrapped_model, (dummy_pos, dummy_node, dummy_connections, dummy_output, dummy_mask))
    print(f"Total FLOPs: {flops.total() / 1e9:.4f} GFLOPs")


    # if config.num_epochs > 0:
    #     best_epoch = train(model,
    #                        train_loader,
    #                        criterion,
    #                        optimizer,
    #                        scheduler,
    #                        device,
    #                        config.num_epochs,
    #                        val_loader,
    #                        test_loader,
    #                        )
    # else:
    #     model.load_state_dict(torch.load(f'{config.ckpt_root}/best_model.pth'))
    #     model = model.to(device)
    #     test(model, test_loader, criterion, device, epoch=0, config=config)