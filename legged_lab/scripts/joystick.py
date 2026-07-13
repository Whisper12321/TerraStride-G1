import time
import pygame
from pygame.locals import *
import queue
import math
import threading

"""
axis:
    0, 左方向摇杆-左右, 范围(-1, 1), cmd_vel_vy
    1, 左方向摇杆-前后, 范围(-1, 1), cmd_vel_vx
    2, 右方向摇杆-前后, 锁定
    3, 右方向摇杆-左右, 范围(-1, 1), cmd_vel_wz

buttons:
    0, A
    1, B
    2, X
    3, Y
    4, LB
    5, RB
"""

class JoyStickController:
    def __init__(self):
        self.joy_freq = 200
        self.running = False
        self.thread = None

        # 数据队列
        self.cmd_vel_queue = queue.Queue(maxsize=5)
        self.button_press_queue = queue.Queue(maxsize=5)

        # 初始化队列数据
        for _ in range(4):
            self.cmd_vel_queue.put(self._init_cmd_vel())
            self.button_press_queue.put(self._init_button_press())

    def _init_button_press(self):
        return {
            "A": 0,
            "B": 0,
            "X": 0,
            "Y": 0,
            "LB": 0,
            "RB": 0
        }

    def _init_cmd_vel(self):
        return {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0
        }

    def start(self):
        """启动手柄线程"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
        self.thread.start()

    def stop(self):
        """停止手柄线程并清理资源"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # 确保pygame正确退出
        try:
            pygame.quit()
        except:
            pass

    def _run(self):
        """线程主循环"""
        try:
            # 在单独的线程中初始化pygame
            pygame.init()
            pygame.joystick.init()

            # 检查并初始化手柄
            joystick_count = pygame.joystick.get_count()
            if joystick_count > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"已连接手柄: {self.joystick.get_name()}")
            else:
                print("未检测到游戏手柄")
                self.running = False
                return

            # 主循环
            while self.running:
                step_start_time = time.time()

                # 更新按钮和摇杆状态
                self._update_button_press()
                self._update_cmd_vel()

                # 控制循环频率
                step_elapsed_time = time.time() - step_start_time
                time.sleep(max(0, 1.0 / self.joy_freq - step_elapsed_time))

        except Exception as e:
            print(f"手柄线程错误: {e}")
        finally:
            # 确保pygame正确退出
            try:
                pygame.quit()
            except:
                pass

    def _update_button_press(self):
        """更新按钮状态（内部使用）"""
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
                return

        # 获取按键值
        if hasattr(self, 'joystick') and self.joystick:
            try:
                button_A = self.joystick.get_button(0)
                button_B = self.joystick.get_button(1)
                button_X = self.joystick.get_button(2)
                button_Y = self.joystick.get_button(3)
                button_LB = self.joystick.get_button(4)
                button_RB = self.joystick.get_button(5)

                new_button_press = {
                    "A": button_A,
                    "B": button_B,
                    "X": button_X,
                    "Y": button_Y,
                    "LB": button_LB,
                    "RB": button_RB
                }

                # 更新队列
                if self.button_press_queue.full():
                    try:
                        self.button_press_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.button_press_queue.put(new_button_press.copy())
            except:
                # 处理手柄断开等异常情况
                pass

    def _update_cmd_vel(self):
        """更新摇杆状态（内部使用）"""
        if hasattr(self, 'joystick') and self.joystick:
            try:
                # 获取摇杆原始值
                axis_1 = self.joystick.get_axis(1)
                axis_0 = self.joystick.get_axis(0)
                axis_3 = self.joystick.get_axis(3)

                # 摇杆速度映射
                vx = (0.65 * (1 - math.cos(math.pi * axis_1)) * (axis_1 >= 0.0) - \
                      0.5 * (1 - math.cos(math.pi * axis_1)) * (axis_1 < 0.0)) * (-1.0)
                vy = (0.5 * (1 - math.cos(math.pi * axis_0)) * (axis_0 >= 0.0) - \
                      0.5 * (1 - math.cos(math.pi * axis_0)) * (axis_0 < 0.0)) * (-1.0)
                wz = (0.5 * (1 - math.cos(math.pi * axis_3)) * (axis_3 >= 0.0) - \
                      0.5 * (1 - math.cos(math.pi * axis_3)) * (axis_3 < 0.0)) * (-1.0)

                new_cmd_vel = {
                    "vx": vx,
                    "vy": vy,
                    "wz": wz
                }

                # 更新队列
                if self.cmd_vel_queue.full():
                    try:
                        self.cmd_vel_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.cmd_vel_queue.put(new_cmd_vel.copy())
            except:
                # 处理手柄断开等异常情况
                pass

    def get_button_press(self):
        """获取最新的按钮状态"""
        try:
            return self.button_press_queue.get_nowait()
        except queue.Empty:
            return self._init_button_press()

    def get_cmd_vel(self):
        """获取最新的摇杆命令"""
        try:
            return self.cmd_vel_queue.get_nowait()
        except queue.Empty:
            return self._init_cmd_vel()

if __name__ == "__main__":
    # 测试代码
    joy = JoyStickController()
    joy.start()

    try:
        while True:
            button = joy.get_button_press()
            cmd_vel = joy.get_cmd_vel()
            print(f"RB: {button['RB']}, Vx: {cmd_vel['vx']:.2f}, Vy: {cmd_vel['vy']:.2f}, Wz: {cmd_vel['wz']:.2f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        joy.stop()
        print("程序已退出")
