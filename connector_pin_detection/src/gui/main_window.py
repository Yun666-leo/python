"""主界面模块"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time


class MainWindow:
    def __init__(self, master, app_controller):
        self.master = master
        self.app = app_controller
        master.title("连接器引脚间距检测判定系统")
        master.geometry("1400x850")
        self._live_preview_active = False
        self._live_thread = None
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

    def _create_menu(self):
        menubar = tk.Menu(self.master)
        fm = tk.Menu(menubar, tearoff=0)
        fm.add_command(label="打开图像", command=self._open_image, accelerator="Ctrl+O")
        fm.add_separator()
        fm.add_command(label="保存报告", command=self._save_report)
        fm.add_separator()
        fm.add_command(label="退出", command=self.master.quit)
        menubar.add_cascade(label="文件", menu=fm)

        cm = tk.Menu(menubar, tearoff=0)
        cm.add_command(label="扫描摄像头", command=self._scan_cameras)
        cm.add_command(label="切换摄像头", command=self._switch_camera_dialog)
        cm.add_command(label="拍照并检测", command=self._camera_capture)
        cm.add_command(label="实时预览", command=self._toggle_live_preview)
        cm.add_separator()
        cm.add_command(label="启动文件夹监听", command=self._start_watch)
        cm.add_command(label="停止文件夹监听", command=self._stop_watch)
        menubar.add_cascade(label="摄像头", menu=cm)

        dm = tk.Menu(menubar, tearoff=0)
        dm.add_command(label="执行检测", command=self._run_detection, accelerator="F5")
        dm.add_command(label="自动标定", command=self._auto_calibrate_dialog)
        dm.add_separator()
        dm.add_command(label="批量处理", command=self._batch_process)
        menubar.add_cascade(label="检测", menu=dm)

        vm = tk.Menu(menubar, tearoff=0)
        vm.add_command(label="显示原图", command=self._show_original)
        vm.add_command(label="显示结果", command=self._show_result_image)
        menubar.add_cascade(label="视图", menu=vm)

        self.master.config(menu=menubar)
        self.master.bind("<Control-o>", lambda e: self._open_image())
        self.master.bind("<F5>", lambda e: self._run_detection())

    def _create_main_layout(self):
        paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        self.view_tabs = ttk.Notebook(left_frame)
        self.view_tabs.pack(fill=tk.BOTH, expand=True)

        img_tab = ttk.Frame(self.view_tabs)
        self.view_tabs.add(img_tab, text="图像")
        self.canvas = tk.Canvas(img_tab, bg="#2b2b2b", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.canvas.bind("<Configure>", lambda e: self._refresh_display())

        preview_tab = ttk.Frame(self.view_tabs)
        self.view_tabs.add(preview_tab, text="实时预览")
        self.preview_canvas = tk.Canvas(preview_tab, bg="#1a1a1a")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)

        info_frame = ttk.LabelFrame(right_frame, text="检测信息/日志")
        info_frame.pack(fill=tk.X, pady=(0, 5))
        self.info_text = tk.Text(info_frame, height=6, state=tk.DISABLED, font=("Microsoft YaHei", 9))
        self.info_text.pack(fill=tk.X, padx=2, pady=2)

        result_frame = ttk.LabelFrame(right_frame, text="判定结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(result_frame, state=tk.DISABLED, font=("Microsoft YaHei", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        btn_frame = ttk.LabelFrame(right_frame, text="操作")
        btn_frame.pack(fill=tk.X)

        # 摄像头选择器
        cam_sel_frame = ttk.Frame(btn_frame)
        cam_sel_frame.pack(fill=tk.X, pady=2)
        ttk.Label(cam_sel_frame, text="摄像头:").pack(side=tk.LEFT, padx=2)
        self.cam_combo = ttk.Combobox(cam_sel_frame, state="readonly", width=18)
        self.cam_combo.pack(side=tk.LEFT, padx=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_cam_selected)
        ttk.Button(cam_sel_frame, text="扫描", command=self._scan_cameras, width=4).pack(side=tk.LEFT, padx=1)

        ttk.Button(btn_frame, text="打开图像", command=self._open_image).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="拍照并检测", command=self._camera_capture).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="实时预览", command=self._toggle_live_preview).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="执行检测(F5)", command=self._run_detection).pack(fill=tk.X, pady=1)
        ttk.Separator(btn_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="📏 自动标定", command=self._auto_calibrate_dialog).pack(fill=tk.X, pady=1)
        ttk.Separator(btn_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="显示原图", command=self._show_original).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="显示结果", command=self._show_result_image).pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame, text="批量处理", command=self._batch_process).pack(fill=tk.X, pady=1)

    def _create_status_bar(self):
        bar = ttk.Frame(self.master)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=1)
        self.cam_status = ttk.Label(bar, text="摄像头: 未连接", foreground="gray", relief=tk.SUNKEN, anchor=tk.E)
        self.cam_status.pack(side=tk.RIGHT, padx=2, pady=1)

    def _scan_cameras(self):
        self.status_var.set("扫描摄像头...")
        self.master.update()
        devices = self.app.camera_list_devices()
        self._cam_map = {}
        labels = []
        for idx, w, h in devices:
            label = f"Camera {idx}  [{w}x{h}]"
            self._cam_map[label] = idx
            labels.append(label)
        self.cam_combo["values"] = labels
        if labels:
            self.cam_combo.current(0)
            self._on_cam_selected()
        self.status_var.set(f"扫描完成: 发现 {len(devices)} 个摄像头")

    def _on_cam_selected(self, event=None):
        label = self.cam_combo.get()
        if label in self._cam_map:
            idx = self._cam_map[label]
            self.app.camera_switch_device(idx)
            self.cam_status.config(text=f"摄像头: Camera {idx}", foreground="blue")
            self._log(f"已切换到 Camera {idx}")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="打开连接器图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")]
        )
        if path:
            try:
                self.app.load_image(path)
                self._display_image(self.app.current_image)
                self.status_var.set(f"已加载: {os.path.basename(path)}")
                self._log(f"加载图像: {os.path.basename(path)}")
                if self.app._auto_detection and self.app.current_image is not None:
                    self._run_detection()
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图像: {str(e)}")

    def _display_image(self, img, canvas=None):
        if canvas is None:
            canvas = self.canvas
        if img is None:
            return
        h, w = img.shape[:2]
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self._pending_display = (img,)
            return
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.delete("all")
        canvas.create_image(cw / 2, ch / 2, anchor=tk.CENTER, image=tk_img)
        canvas.image = tk_img

    def _refresh_display(self):
        if hasattr(self, "_pending_display"):
            self._display_image(*self._pending_display)
            del self._pending_display
        elif self.app.current_image is not None:
            self._display_image(self.app.current_image)

    def _show_original(self):
        if self.app.current_image is not None:
            self._display_image(self.app.current_image)

    def _show_result_image(self):
        if self.app.result_image is not None:
            self._display_image(self.app.result_image)

    def _run_detection(self):
        if self.app.current_image is None:
            messagebox.showinfo("提示", "请先打开图像或拍照")
            return
        self.status_var.set("检测中...")
        self.master.update()
        result = self.app.run_detection()
        if result:
            self._display_image(self.app.result_image)
            self._show_result(result)
            self._log("检测完成")
        else:
            self._log("检测失败或引脚不足")
        self.status_var.set("就绪")

    def _show_result(self, result):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        # 引脚坐标
        if self.app.detection_result and self.app.detection_result.pins:
            self.result_text.insert(tk.END, "引脚坐标:\n")
            for p in self.app.detection_result.pins:
                self.result_text.insert(tk.END, f"  Pin{p.index}: ({p.center[0]:.0f},{p.center[1]:.0f})\n")
            self.result_text.insert(tk.END, "\n")
        self.result_text.insert(tk.END, f"整体判定: {result.overall_verdict}\n")
        self.result_text.insert(tk.END, f"{result.message}\n\n")
        if result.pitch_stats:
            self.result_text.insert(tk.END, "间距统计:\n")
            for k, v in result.pitch_stats.items():
                self.result_text.insert(tk.END, f"  {k}: {v}\n")
        if result.pin_details:
            ok = sum(1 for d in result.pin_details if d[2] == "合格")
            ng = len(result.pin_details) - ok
            self.result_text.insert(tk.END, f"\n引脚判定: 合格{ok} / 异常{ng}\n")
        self.result_text.config(state=tk.DISABLED)

    def _log(self, msg):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, time.strftime("%H:%M:%S ") + msg + "\n")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

    # ========== 摄像头操作 ==========

    def _switch_camera_dialog(self):
        """弹出切换对话框"""
        dialog = tk.Toplevel(self.master)
        dialog.title("切换摄像头")
        dialog.geometry("300x200")
        ttk.Label(dialog, text="选择摄像头设备:").pack(pady=10)
        lb = tk.Listbox(dialog, height=8)
        lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        devices = self.app.camera_list_devices()
        for idx, w, h in devices:
            lb.insert(tk.END, f"Camera {idx}  [{w}x{h}]")

        def on_select():
            sel = lb.curselection()
            if sel:
                idx = devices[sel[0]][0]
                self.app.camera_switch_device(idx)
                self.cam_status.config(text=f"摄像头: Camera {idx}", foreground="blue")
                self._log(f"已切换到 Camera {idx}")
            dialog.destroy()

        ttk.Button(dialog, text="切换", command=on_select).pack(pady=5)

    def _camera_capture(self):
        self.status_var.set("拍照中...")
        self.master.update()
        if not self.app.camera.is_opened:
            if not self.app.camera_open_live():
                messagebox.showerror("错误", f"无法打开摄像头: {self.app.camera.last_error}")
                self.status_var.set("拍照失败")
                return
        ok = self.app.camera_capture()
        if ok:
            self._display_image(self.app.current_image)
            if self.app.classification_result:
                self._show_result(self.app.classification_result)
            self._log(f"拍照成功: {self.app.current_source}")
            self.status_var.set("拍照完成")
        else:
            messagebox.showerror("错误", "拍照失败")
            self.status_var.set("拍照失败")

    def _toggle_live_preview(self):
        if self._live_preview_active:
            self._stop_live_preview()
        else:
            self._start_live_preview()

    def _start_live_preview(self):
        if not self.app.camera.is_opened:
            if not self.app.camera.open():
                messagebox.showerror("错误", f"无法打开摄像头: {self.app.camera.last_error}")
                return
        self._live_preview_active = True
        self.cam_status.config(text="摄像头: 预览中", foreground="green")
        self.view_tabs.select(1)
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()
        self._log("实时预览已启动")

    def _stop_live_preview(self):
        self._live_preview_active = False
        if self._live_thread:
            self._live_thread.join(timeout=1)
            self._live_thread = None
        self.cam_status.config(text="摄像头: 已连接", foreground="blue")
        self._log("实时预览已停止")

    def _live_loop(self):
        while self._live_preview_active and self.app.camera.is_opened:
            frame = self.app.camera.read_frame()
            if frame and frame.image is not None:
                self.master.after(0, lambda: self._display_image(frame.image, self.preview_canvas))
            time.sleep(0.03)

    # ========== 自动标定 ==========

    def _auto_calibrate_dialog(self):
        """弹出输入标称间距的对话框，执行自动标定"""
        if not self.app.pitches or len(self.app.pitches) < 2:
            messagebox.showwarning("提示", "请先执行检测，检测到足够引脚后再进行自动标定")
            return

        pitch = simpledialog.askfloat(
            "自动标定",
            "输入连接器的标称引脚间距 (mm)\n\n"
            f"当前检测到 {len(self.app.pitches)} 组相邻引脚\n"
            f"像素间距均值: {np.mean([p[2] for p in self.app.pitches]):.1f} px\n\n"
            "示例: 1.27, 1.0, 2.54, 0.5",
            parent=self.master,
            minvalue=0.001,
            maxvalue=100.0
        )
        if pitch is None:
            return  # 用户取消了

        self.status_var.set("标定中...")
        self.master.update()
        result = self.app.auto_calibrate(pitch)
        if result["success"]:
            messagebox.showinfo(
                "标定完成",
                f"✅ 自动标定成功！\n\n"
                f"标称间距: {result['nominal_pitch_mm']} mm\n"
                f"平均像素间距: {result['avg_pixel_distance']} px\n"
                f"新 pixel_per_mm: {result['pixel_per_mm']}\n"
                f"参与引脚组数: {result['pitch_count']}\n\n"
                f"配置已保存到 default_config.yaml"
            )
            self._log(f"自动标定: {result['message']}")
            # 用新标定结果刷新显示
            if self.app.result_image is not None:
                self._display_image(self.app.result_image)
            if self.app.classification_result:
                self._show_result(self.app.classification_result)
            self.status_var.set(f"标定完成: ppm={result['pixel_per_mm']}")
        else:
            messagebox.showerror("标定失败", result["message"])
            self.status_var.set("标定失败")

    # ========== 监听 ==========

    def _start_watch(self):
        folder = filedialog.askdirectory(title="选择监听文件夹")
        if folder:
            self.app.start_watch(folder, on_result=self._on_watch_result)
            self._log(f"文件夹监听已启动: {folder}")
            self.status_var.set(f"监听中: {folder}")
            self.cam_status.config(text="监听中", foreground="green")

    def _stop_watch(self):
        self.app.stop_watch()
        self._log("文件夹监听已停止")
        self.status_var.set("就绪")
        self.cam_status.config(text="摄像头: 未连接", foreground="gray")

    # ========== 批量 / 保存 ==========

    def _on_watch_result(self):
        """监听检测完成后刷新 GUI 显示（在 watcher 后台线程回调，需切回主线程）"""
        self.master.after(0, self._refresh_watch_display)

    def _refresh_watch_display(self):
        """主线程中更新界面"""
        self._display_image(self.app.result_image if self.app.result_image is not None else self.app.current_image)
        if self.app.classification_result:
            self._show_result(self.app.classification_result)
        self._log("监听检测完成")
        self.status_var.set("监听检测完成")

    def _batch_process(self):
        folder = filedialog.askdirectory(title="选择批量处理文件夹")
        if folder:
            self.status_var.set("批量处理中...")
            self.master.update()
            report = self.app.batch_process(folder)
            self._log(f"批量完成: 处理 {len(report)} 个文件")
            self.status_var.set("批量处理完成")

    def _save_report(self):
        path = filedialog.asksaveasfilename(
            title="保存检测报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if path:
            self.app.save_report(path)
            self._log(f"报告已保存: {os.path.basename(path)}")
            self.status_var.set("报告已保存")
