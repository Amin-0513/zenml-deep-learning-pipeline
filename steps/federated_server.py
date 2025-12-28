import socket
from zenml import step

@step()
def connect_step(host: str, port: int)-> socket.socket:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    msg = client_socket.recv(1024).decode()
    print("Server:", msg)
    return client_socket
