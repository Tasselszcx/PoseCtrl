import socket
import json
import io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class UnityController:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port

    def _recvall(self, sock, n):
        """可靠接收指定字节数"""
        data = bytearray()
        while len(data) < n:
            remaining = n - len(data)
            packet = sock.recv(4096 if remaining > 4096 else remaining)
            if not packet:
                raise ConnectionError("Connection closed prematurely")
            data.extend(packet)
        return bytes(data)

    def send_command(self, command_type, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            
            # 发送指令
            s.sendall(json.dumps({
                "type": command_type,
                "data": data
            }).encode('utf-8'))

            # 渲染命令需要接收图像数据
            if command_type == "render":
                # 读取4字节长度头
                length_bytes = self._recvall(s, 4)
                length = int.from_bytes(length_bytes, byteorder='little')
                
                # 读取图像数据
                image_data = self._recvall(s, length)
                return Image.open(io.BytesIO(image_data))
            else:
                return None

    def Control_render(
        self,
        x, y, z,
        x_r, y_r, z_r,
        fov,
        width=1024,
        height=1024
    ) -> Image.Image:
        """返回渲染的原始图像（未经过姿势处理）"""
        # 设置相机参数
        self.send_command("camera", {
            "x": x,
            "y": y,
            "z": z,
            "rx": x_r,
            "ry": y_r,
            "rz": z_r,
            "fov": fov
        })

        # 触发渲染并获取图像
        return self.send_command("render", {
            "width": width,
            "height": height
        })

