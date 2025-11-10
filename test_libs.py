# import torch
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

# import torch
# print(torch.cuda.get_device_name(0))
# x = torch.randn(1024, 1024, device="cuda")
# y = torch.mm(x, x)
# torch.cuda.synchronize()
# print("OK", y.mean().item())
#
# torch.backends.cudnn.benchmark = True
# print(torch.cuda.mem_get_info())
# torch.cuda.reset_peak_memory_stats()
