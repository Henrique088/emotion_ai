from dataset import FERDataset

ds = FERDataset("../data/fer2013.csv")

img, label = ds[0]

print(img.shape)
print(label)
