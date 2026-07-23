import os
import time
import numpy as np
import random
import sys
import yaml
from utility import last_converted_new, generate_run_in, prepare_initial_model_xyz
import subprocess

#============== 配置参数区 ================
# 加载YAML配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

Dirsstopsteps = config['Dirsstopsteps']         # 迭代停止的步数
substrate_temperature = config['substrate_temperature']     # 衬底温度 (K)
inject_temperature = config['inject_temperature']       # 入射粒子温度 (K)
injection_flux = config.get('injection_flux', 1.0e-7)  # 兼容旧配置：未设置时仅作回退
inject_xy_spacing = config.get('inject_xy_spacing', 8.0)  # XY 网格间距（Å）
injection_count = config.get('injection_count')           # 每轮入射粒子数（优先于网格满铺）
z_spread = config.get('z_spread', 50.0)                   # 腔体 Z 均匀分布宽度（Å）
local_surface_radius = config.get('local_surface_radius', 5.0)
surface_height_offset = config.get('surface_height_offset', 15.0)
cluster_cutoff = config.get('cluster_cutoff', 2.8)
remove_incident_particles = config.get('remove_incident_particles', True)
velocity_magnitude = config.get('velocity_magnitude')       # 入射速度模长（Å/fs），优先于 inject_temperature
theta_sigma_deg = config.get('theta_sigma_deg', 5.0)      # 束流发散角 σ_θ（度），高斯仅影响速度方向
time_step = config['time_step']                    # 时间步长 (fs)
run_steps = config['run_steps']                # 每次迭代运行步数
initial_xyz = config['initial_xyz']            # 初始衬底输入文件
substrate_replicate = config.get('substrate_replicate')  # XY 复制倍数，仅首轮生效
box_z = config['box_z']                       # box的Z高度 (Å)
gpu_device = config.get('gpu_device', 0)              # 批量任务由 run_batch 写入；单跑时默认 0
if 'BATCH_GPU_ID' in os.environ:
    gpu_device = int(os.environ['BATCH_GPU_ID'])   # 批量并行时由调度器指定
gpumd_command = config['gpumd_command']    # GPUMD执行命令
inject_species_weights = config.get('inject_species_weights', {'Ge': 0.3, 'Si': 0.7})  # 入射粒子物种权重
enable_d3 = config.get('enable_d3', False)  # d3 控件开关（默认关闭）

#=========================================




start_time = time.time()  # 获取当前时间
print(f"The program starts at {start_time} seconds.")

while True:
    current_directory = os.getcwd() #获取当前目录
    all_entries = os.listdir(current_directory)  # 获取程序所在目录下已存在的子目录
    existing_directories = [entry for entry in all_entries if os.path.isdir(os.path.join(current_directory, entry))]  # 获取程序所在目录下已存在的子目录
    existing_directories.remove('__pycache__') if '__pycache__' in existing_directories else existing_directories
    # 将一些准备文件给删除，比如0开头的一些POSCAR的准备文件夹
    existing_directories=[i for i in existing_directories if not i.startswith ('0')]
    existing_directories.remove('nep') if 'nep' in existing_directories else existing_directories
    print(" 已存在的子文件夹 ",existing_directories)
    
    if len(existing_directories)==Dirsstopsteps:
        end_time = time.time()
        run_time = end_time - start_time
        print("已在此文件夹:", current_directory, "下执行成功", Dirsstopsteps, "个子文件夹退出程序")
        print(f"Total wall time: {run_time:.1f} s ({run_time / 3600:.2f} h)")
        sys.exit()
        
    try:
        end_time = time.time()  # 获取程序结束后的时间
        run_time = end_time - start_time

        
        old_directory=os.path.join(current_directory,str(len(existing_directories)))
        old_dumpxyz=os.path.join(old_directory,'dump.xyz')
        

        if len(existing_directories) == 0 and os.path.exists(initial_xyz):
            # 首次运行，从初始文件创建目录1（substrate_replicate 仅在此步生效）
            new_directory = os.path.join(current_directory, '1')
            os.makedirs(new_directory,exist_ok=True)
            model_xyz = os.path.join(new_directory, 'model.xyz')
            rep_info = prepare_initial_model_xyz(
                initial_xyz,
                model_xyz,
                substrate_replicate=substrate_replicate,
            )
            if rep_info.get("replicated"):
                print(
                    f"Substrate XY replicate {rep_info['nx']}x{rep_info['ny']}: "
                    f"{rep_info['n_in']} -> {rep_info['n_out']} atoms"
                )
        else:
            new_directory = os.path.join(current_directory, str(len(existing_directories)+1))
            os.makedirs(new_directory,exist_ok=True)
            print("new_directory",new_directory)
            # 修改生产新的xyz，设置入射粒子温度
            new_modelxyz = os.path.join(new_directory, 'model.xyz')
            print("new  modelxyz",new_modelxyz)
            last_converted_new(
                file_path=old_dumpxyz,
                output_file=new_modelxyz,
                inject_temperature=inject_temperature,
                injection_flux=injection_flux,
                box_z=box_z,
                time_step=time_step,
                run_steps=run_steps,
                inject_species_weights=inject_species_weights,
                inject_xy_spacing=inject_xy_spacing,
                injection_count=injection_count,
                z_spread=z_spread,
                local_surface_radius=local_surface_radius,
                surface_height_offset=surface_height_offset,
                cluster_cutoff=cluster_cutoff,
                remove_incident_particles=remove_incident_particles,
                velocity_magnitude=velocity_magnitude,
                theta_sigma_deg=theta_sigma_deg,
            )
            # 检查并替换 model.xyz 中的 nan 为 0.0
            subprocess.run(
                ['sed', '-i', 's/nan/0.0/g', new_modelxyz],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ['sed', '-i', 's/-nan/0.0/g', new_modelxyz],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        
        # 生成 run.in 文件，设置衬底温度
        generate_run_in(
            new_directory,
            substrate_temperature,
            time_step,
            run_steps,
            enable_d3=enable_d3,
        )

        # 提交GPUMD任务
        subprocess.run(f"cd {new_directory};CUDA_VISIBLE_DEVICES={gpu_device} {gpumd_command}", shell=True, check=True)


    except Exception as e:
        end_time = time.time()  # 获取程序结束后的时间
        run_time = end_time - start_time
        print(f"程序运行出错: {e}")
        print(f"The program ran for {run_time} seconds.")    
