# steps/close_step.py
from zenml import step

@step
def close_step(client_socket):
    client_socket.sendall(b"close")
    client_socket.close()
    print("Connection closed")
