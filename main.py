import torch

def main():
    print("PyTorch安装路径:", torch.__file__)
    print("是否支持CUDA:", torch.cuda.is_available())
    print("CUDA架构列表:", torch.cuda.get_arch_list())
    print("Hello from minimind!")


if __name__ == "__main__":
    main()
