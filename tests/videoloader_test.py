from utils import VideoLoader

loader = VideoLoader(224, 18, 1)

vt, vm, at = loader.load(".assets/dia0_utt0.mp4")

print(vt)
print(vt.shape)
print(vm)
print(at)
print(at.shape)
