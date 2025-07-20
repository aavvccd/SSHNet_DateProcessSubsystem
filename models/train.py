import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from Model import SSHCompensationNet
from train_dataloader import OceanDataProcess
from Lossfun import masked_mse_loss
# 训练函数
def train_model():
    # 超参数配置
    config = {
        'batch_size': 4,
        'learning_rate': 2e-4,
        'num_epochs': 200,
        'patch_size': 32,
        'data_paths': {
            'merged': r"E:\1m\src\merged_grid_0110.nc",
            'ssh': r"E:\1m\src\ssh_interpolated.nc",
            'sst': r"E:\1m\src\sst_interpolated.nc",
            'wind': r"E:\1m\src\wind_interpolated.nc"
        },
        'save_dir': 'checkpoints',
        'results_dir': 'results'
    }

    # 创建目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    # 创建时间戳标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 加载数据集
    print("加载数据集...")
    dataset = OceanDataProcess(
        merged_path=config['data_paths']['merged'],
        ssh_path=config['data_paths']['ssh'],
        sst_path=config['data_paths']['sst'],
        wind_path=config['data_paths']['wind'],
        patch_size=config['patch_size']
    )

    norm_params = dataset.get_norm_params()
    norm_params_path = os.path.join(config['save_dir'], f'norm_params_{timestamp}.pkl')
    with open(norm_params_path, 'wb') as f:
        pickle.dump(norm_params, f)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"使用设备: {device}")

    model = SSHCompensationNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    train_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"训练周期 {epoch + 1}/{config['num_epochs']}")

        for sat_all, real_ssh, mask in progress_bar:
            sat_all = sat_all.to(device)
            real_ssh = real_ssh.to(device)
            mask = mask.to(device)

            sat_measurements = sat_all[:, :15, :, :]
            sat_ssh = sat_all[:, 15:16, :, :]
            env_features = sat_all[:, 16:19, :, :]
            true_compensation = real_ssh - sat_ssh
            print(sat_ssh.min(), sat_ssh.max())
            print(real_ssh.min(), real_ssh.max())
            pred_compensation = model(sat_measurements, env_features)
            loss = masked_mse_loss(pred_compensation, true_compensation, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})

        train_loss /= num_batches

        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for sat_all, real_ssh, mask in val_loader:
                sat_all = sat_all.to(device)
                real_ssh = real_ssh.to(device)
                mask = mask.to(device)
                # 拆分输入数据
                sat_measurements = sat_all[:, :15, :, :]
                sat_ssh = sat_all[:, 15:16, :, :]
                env_features = sat_all[:, 16:19, :, :]
                print(sat_ssh.min(),sat_ssh.max())
                print(real_ssh.min(),real_ssh.max())
                true_compensation = real_ssh - sat_ssh
                pred_compensation = model(sat_measurements, env_features)
                loss = masked_mse_loss(pred_compensation, true_compensation, mask)
                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= num_val_batches

        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)
        train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config,
                'norm_params': norm_params  # 保存归一化参数
            }, os.path.join(config['save_dir'], f'best_model_{timestamp}.pth'))
        print(f"\n周期 {epoch + 1}/{config['num_epochs']} - "
              f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, "
              f"学习率: {optimizer.param_groups[0]['lr']:.2e}")

    # 保存最终模型
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        'config': config,
        'norm_params': norm_params  # 保存归一化参数
    }, os.path.join(config['save_dir'], f'final_model_{timestamp}.pth'))


    plot_loss_curves(train_history, config['results_dir'], timestamp)

    return model, train_history


def plot_loss_curves(history, save_dir, timestamp):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='训练损失')
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='验证损失')
    plt.title('val and train loss')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['learning_rate'], 'g-', label='学习率')
    plt.title('lr')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plot_path = os.path.join(save_dir, f'loss_curves_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存至: {plot_path}")

    csv_path = os.path.join(save_dir, f'training_history_{timestamp}.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,learning_rate\n")
        for i in range(len(history['epoch'])):
            f.write(
                f"{history['epoch'][i]},{history['train_loss'][i]},{history['val_loss'][i]},{history['learning_rate'][i]}\n")
    print(f"训练历史已保存至: {csv_path}")


def test_model(model_path, test_dataset, device='cuda'):
    model = SSHCompensationNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_params = checkpoint.get('norm_params', None)
    if norm_params is None:
        norm_params = {
            'sat_means': [0.0] * 18,
            'sat_stds': [1.0] * 18
        }

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    all_pred_ssh = []
    all_true_ssh = []
    all_sat_ssh = []
    all_pred_compensation = []

    with torch.no_grad():
        for sat_all, real_ssh, mask in test_loader:
            sat_all = sat_all.to(device)
            real_ssh = real_ssh.to(device)
            mask = mask.to(device)

            sat_measurements = sat_all[:, :15, :, :]
            sat_ssh = sat_all[:, 15:16, :, :]
            env_features = sat_all[:, 16:19, :, :]

            pred_compensation = model(sat_measurements, env_features)

            pred_ssh = sat_ssh + pred_compensation

            all_pred_ssh.append(pred_ssh.cpu())
            all_true_ssh.append(real_ssh.cpu())
            all_sat_ssh.append(sat_ssh.cpu())
            all_pred_compensation.append(pred_compensation.cpu())

    pred_ssh = torch.cat(all_pred_ssh, dim=0)
    true_ssh = torch.cat(all_true_ssh, dim=0)
    sat_ssh = torch.cat(all_sat_ssh, dim=0)
    pred_compensation = torch.cat(all_pred_compensation, dim=0)

    original_error = torch.abs(sat_ssh - true_ssh)
    original_mae = original_error.mean().item()

    corrected_error = torch.abs(pred_ssh - true_ssh)
    corrected_mae = corrected_error.mean().item()

    compensation_error = torch.abs(pred_compensation - (true_ssh - sat_ssh))
    compensation_mae = compensation_error.mean().item()

    improvement = (original_mae - corrected_mae) / original_mae * 100

    print("\n测试结果:")
    print("=" * 50)
    print(f"原始SSH MAE: {original_mae:.6f} m")
    print(f"修正后SSH MAE: {corrected_mae:.6f} m")
    print(f"补偿值预测MAE: {compensation_mae:.6f} m")
    print(f"相对改进: {improvement:.2f}%")
    print("=" * 50)

    return {
        'original_mae': original_mae,
        'corrected_mae': corrected_mae,
        'compensation_mae': compensation_mae,
        'improvement': improvement
    }


if __name__ == "__main__":
    trained_model, history = train_model()

    best_model_path = os.path.join('checkpoints', 'best_model_*.pth')  # 替换为实际路径
    checkpoint = torch.load(best_model_path)
    norm_params = checkpoint.get('norm_params', None)

    test_dataset = OceanDataProcess(
        merged_path=r"E:\1m\src\merged_grid_0110.nc",
        ssh_path=r"E:\1m\src\ssh_interpolated.nc",
        sst_path=r"E:\1m\src\sst_interpolated.nc",
        wind_path=r"E:\1m\src\wind_interpolated.nc",
        patch_size=32,
        norm_params=norm_params
    )

    test_results = test_model(best_model_path, test_dataset)

    results_path = os.path.join('results', 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("测试结果:\n")
        f.write("=" * 50 + "\n")
        f.write(f"原始SSH MAE: {test_results['original_mae']:.6f} m\n")
        f.write(f"修正后SSH MAE: {test_results['corrected_mae']:.6f} m\n")
        f.write(f"补偿值预测MAE: {test_results['compensation_mae']:.6f} m\n")
        f.write(f"相对改进: {test_results['improvement']:.2f}%\n")
        f.write("=" * 50 + "\n")

    print(f"测试结果已保存至: {results_path}")
