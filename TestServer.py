import random
import socket
import time

HOST = "127.0.0.1"
PORT = 2063

class Points:
    def __init__(self, size):
        self.size = size
    
    def rawPoints(self):
        points = []
        for i in range(self.size):
            x = random.randint(-9, 9) + random.random()
            y = random.randint(-9, 9) + random.random()
            x = round(x, 3)
            y = round(y, 3)
            points.append([x, y])
        return points

    def encodedPoints(self):
        points = self.rawPoints()
        pointStr = ""
        for i in range(self.size):
            x = str(points[i][0])
            y = str(points[i][1])
            pointStr += "," + x + "," + y
        return pointStr


def pointsFromInput(n):
    return Points(n).encodedPoints()

# # Template Server
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print(f"Connected by {addr}")
#         while True:
#             data = conn.recv(1024)
#             if not data:
#                 break
#             data = data.decode()
#             # Do whatever with data
#             conn.sendall()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            n = int(data.decode())
            print(f"Received request for {n} waypoints")
            points = pointsFromInput(n)
            print("Sending: " + points)
            conn.sendall(points.encode())