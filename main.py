import os
import gc
import numpy as np
from lib.NcOp import NcFileIO
from lib.dataprocess import datarepair
from lib.dataprocess import oldfilter
from scipy.interpolate import griddata
from lib.dataprocess.waveform import WaveformReshaper
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

resolution = 1  # 生成网格的分辨率
intermathord = 'linear'


class DataProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("卫星数据处理工具")
        master.geometry("800x600")

        # 输入输出目录框架
        self.setup_path_controls()

        # 控制台输出区域
        self.setup_console()

        # 控制按钮
        self.setup_buttons()

        # 队列和状态管理
        self.queue = queue.Queue()
        self.running = False
        self.master.after(100, self.check_queue)

    def setup_path_controls(self):
        """路径选择控件"""
        frame = ttk.Frame(self.master)
        frame.pack(pady=10, fill=tk.X)

        # 输入目录
        ttk.Label(frame, text="输入目录:").grid(row=0, column=0, padx=5)
        self.entry_input = ttk.Entry(frame, width=40)
        self.entry_input.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="浏览...", command=self.select_input).grid(row=0, column=2, padx=5)

        # 输出目录
        ttk.Label(frame, text="输出目录:").grid(row=1, column=0, padx=5)
        self.entry_output = ttk.Entry(frame, width=40)
        self.entry_output.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="浏览...", command=self.select_output).grid(row=1, column=2, padx=5)

    def setup_console(self):
        """控制台输出区域"""
        console_frame = ttk.Frame(self.master)
        console_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            wrap=tk.WORD,
            state='disabled',
            font=('Consolas', 10)
        )
        self.console.pack(fill=tk.BOTH, expand=True)

    def setup_buttons(self):
        """操作按钮"""
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(pady=10)

        self.btn_start = ttk.Button(btn_frame, text="开始处理", command=self.start_processing)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_clear = ttk.Button(btn_frame, text="清空控制台", command=self.clear_console)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

    def select_input(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, path)

    def select_output(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)

    def start_processing(self):
        if self.running:
            messagebox.showwarning("警告", "处理正在进行中")
            return

        input_dir = self.entry_input.get()
        output_dir = self.entry_output.get()

        if not self.validate_paths(input_dir, output_dir):
            return

        self.running = True
        self.btn_start.config(state=tk.DISABLED)

        threading.Thread(
            target=self.process_data,
            args=(input_dir, output_dir),
            daemon=True
        ).start()

    def validate_paths(self, input_dir, output_dir):
        if not os.path.isdir(input_dir):
            messagebox.showerror("错误", "输入目录无效")
            return False
        if not os.path.isdir(output_dir):
            messagebox.showerror("错误", "输出目录无效")
            return False
        return True

    def process_data(self, input_dir, output_dir):
        try:
            dirs = os.listdir(input_dir)
            total = len(dirs)

            self.queue_message(f"找到 {total} 个周期目录")
            self.queue_message("=" * 50)

            for idx, cycle in enumerate(dirs, 1):
                try:
                    self.process_cycle(input_dir, output_dir, cycle, idx, total)
                except Exception as e:
                    self.queue_message(f"[错误] 处理 {cycle} 失败: {str(e)}", "red")
                    break

            self.queue.put(("done", "处理完成！"))
        except Exception as e:
            self.queue.put(("error", f"初始化错误: {str(e)}"))

    def process_cycle(self, input_dir, output_dir, cycle, idx, total):
        """示例处理流程（需替换为实际逻辑）"""
        self.queue_message(f"正在处理周期 {cycle} ({idx}/{total})", "blue")

        # cydle子目录创建
        data_out_dirname = os.path.join(output_dir, cycle)
        datarepair.create_folder(data_out_dirname)
        self.queue_message(f"√ 创建输出子目录{data_out_dirname}", "green")

        file_rootdir = input_dir
        # 模拟处理步骤
        self.queue_message(f"1. 读取 {cycle} 数据...")
        # 这里添加实际的数据读取代码
        gridded_data = {}
        file_datadir = os.path.join(file_rootdir, cycle)
        nc_filedata = NcFileIO.read_netcdf4(file_datadir, 'r')
        data_arrays = {  # 数据标签
            'lon': [],
            'lat': [],
            'surface_type': [],
            'mean_sea_surface': [],
            'range_ku': [],
            'swh_ku': [],
            'sig0_ku': [],
            'alt': [],
            'wind_speed_model_u': [],
            'wind_speed_model_v': [],
            'wind_speed_alt': [],
            'wind_speed_rad': [],
            'agc_ku': [],
            'agc_numval_ku': [],
            'rad_water_vapor': [],
            'off_nadir_angle_wf_ku': [],
            'geoid': [],
            # 'waveforms_20hz_ku': []  # 二维数据可用嵌套列表（如 [[v1, v2, ...], ...]）
        }
        time = []
        waveforms_20hz_ku = []
        qual_data_arrays = {
            'qual_alt_1hz_range_ku': [],
            'qual_alt_1hz_swh_ku': [],
            'qual_alt_1hz_sig0_ku': [],
            # 可添加更多
        }
        data_lable_list = list(data_arrays.keys())
        qual_lable_list = list(qual_data_arrays.keys())
        for nc_data in nc_filedata:
            waveforms_20hz_ku.append(nc_data.variables['waveforms_20hz_ku'][:])
            time.append(nc_data.variables['time'][:])
            for lable in data_lable_list:
                data_arrays[lable].append(nc_data.variables[lable][:])
            for lable in qual_lable_list:
                qual_data_arrays[lable].append(nc_data.variables[lable][:])
        for lable in data_lable_list:
            data_arrays[lable] = np.concatenate(data_arrays[lable])
        for lable in qual_lable_list:
            qual_data_arrays[lable] = np.concatenate(qual_data_arrays[lable])
            qual_data_arrays[lable] = np.where(qual_data_arrays[lable] == 1, False, True)
        waveforms_20hz_ku = np.concatenate(waveforms_20hz_ku)
        time = np.concatenate(time)
        self.queue_message(f"√ 数据读取完成 ({cycle})", "green")

        self.queue_message(f"2. 处理波形数据...")
        # 这里添加实际的波形处理代码
        # 初始化波形处理器
        self.queue_message(f"   初始化波形处理器", "green")
        reshaper = WaveformReshaper(waveforms_20hz_ku.shape[0], 20, 128)
        self.queue_message(f"   初始化完成", "green")
        reshaped_data = reshaper.reshape_to_3d(waveforms_20hz_ku)
        reshaper.import_to_nc(data_out_dirname, reshaped_data, time)  # 导出波形到文件
        del reshaper
        del reshaped_data
        self.queue_message(f"√ 波形处理完成 ({cycle})，并保存到(waveform_3d_netcdf4.nc)", "green")

        self.queue_message(f"3. 处理其他数据...")
        lon_all = data_arrays['lon']
        lat_all = data_arrays['lat']
        lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
        lon_min, lon_max = np.floor(lon_all.min()), np.ceil(lon_all.max())
        lat_min, lat_max = np.floor(lat_all.min()), np.ceil(lat_all.max())
        grid_lon = np.arange(lon_min, lon_max + resolution, resolution)
        grid_lat = np.arange(lat_min, lat_max + resolution, resolution)
        grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
        for lable in data_lable_list:
            if (lable == 'lat') or (lable == 'lon'):
                continue
            self.queue_message(f"   处理数据 ({lable})", "blue")
            base_mask = (data_arrays['surface_type'] != 2) & (data_arrays['surface_type'] != 3)
            if (lable != 'surface_type'):
                qual = datarepair.qul_control(data_arrays[lable], lable, qual_lable_list, qual_data_arrays)
                mask = base_mask & qual
            else:
                mask = base_mask

            mask_lon = lon_all[mask]
            mask_lat = lat_all[mask]

            current_data = data_arrays[lable][mask]
            points = np.column_stack((mask_lon, mask_lat))

            if lable != 'surface_type':
                grid_data = griddata(
                    points, current_data,
                    (grid_lon2d, grid_lat2d),
                    method=intermathord,
                    fill_value=np.nan
                )
            else:
                grid_data = griddata(
                    points, current_data,
                    (grid_lon2d, grid_lat2d),
                    method='nearest',
                    fill_value=np.nan
                )
            # 存储插值结果到字典
            grid_data = oldfilter.sliding_window_filter(grid_data, 3, 'gaussian')  # 滤波
            if lable == 'surface_type':
                grid_data = np.ceil(grid_data)
            gridded_data[lable] = grid_data
            datarepair.plot(data_out_dirname, resolution, lable, grid_lon2d, grid_lat2d, grid_data)
        self.queue_message(f"√ 其他数据处理完成 ({cycle})", "green")


        self.queue_message(f"4. 保存处理结果...")
        # 这里添加实际的保存代码
        output_path = os.path.join(data_out_dirname, f"merged_grid_{cycle}.nc")
        self.queue_message(f"生成文件(merged_grid_{cycle}.nc)")
        datarepair.export_to_netcdf(gridded_data, grid_lon, grid_lat, output_path, resolution)
        self.queue_message(f"√ 结果保存完成 ({cycle})", "green")

        del nc_filedata  # 删除文件数据对象
        del data_arrays  # 删除原始数据集合
        del qual_data_arrays  # 删除质量控制数据
        del waveforms_20hz_ku  # 删除波形原始数据
        del time  # 删除时间数据
        del grid_lon, grid_lat, grid_lon2d, grid_lat2d  # 删除网格坐标
        del gridded_data  # 删除插值后的网格数据
        del data_out_dirname, file_datadir  # 重置目录
        gc.collect()  # 强制垃圾回收

        self.queue_message("-" * 50)

    def queue_message(self, message, color="black"):
        """将消息加入队列"""
        self.queue.put(("message", message, color))

    def clear_console(self):
        """清空控制台"""
        self.console.config(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.config(state='disabled')

    def check_queue(self):
        """处理队列消息"""
        try:
            while True:
                msg = self.queue.get_nowait()

                if msg[0] == "message":
                    self.append_to_console(msg[1], msg[2])
                elif msg[0] == "done":
                    messagebox.showinfo("完成", msg[1])
                    self.running = False
                    self.btn_start.config(state=tk.NORMAL)
                elif msg[0] == "error":
                    messagebox.showerror("错误", msg[1])
                    self.running = False
                    self.btn_start.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        self.master.after(100, self.check_queue)

    def append_to_console(self, text, color="black"):
        """向控制台添加带颜色的文本"""
        self.console.config(state='normal')
        self.console.insert(tk.END, text + "\n")
        self.console.tag_add(color, "end-2l linestart", "end-1c")
        self.console.tag_config(color, foreground=color)
        self.console.see(tk.END)
        self.console.config(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = DataProcessorApp(root)
    root.mainloop()
