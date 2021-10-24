import tensorflow_datasets as tfds

# dataset, ds_info = tfds.load(name='mnist',
#                                   shuffle_files=True,
#                                   with_info=True)
#
# # n_train = ds_info.splits['train'].num_examples
# # n_test = ds_info.splits['test'].num_examples
#
# # print(ds_info)
# # print(ds_info.features)
# # print(ds_info.splits)
#
# dataset = tfds.load(name='mnist',
#                     shuffle_files=True)
#
# # print(type(dataset))
# # print(dataset.keys(),'\n')
# # print(dataset.values())
#
# train_ds = dataset['train'].batch(32)
# test_ds = dataset['test']
#
# # print(type(train_ds))
# # print(type(test_ds))
#
# for epoch in range(EPOCHS):
#     for data in train_ds :
#         images = data['image']
#         labels = data['label']
#
# for tmp in train_ds :
#     print(type(tmp))
#
#     print(tmp.keys())
#
#     images = tmp['image']
#     labels = tmp['label']
#
#     print(images.shape)
#     print(labels.shape)
#
#     break


# #as_supervised 할 때 tuple 형태로 담아줌
# dataset = tfds.load(name='mnist',
#                     shuffle_files=True,
#                     as_supervised=True)
#
# train_ds = dataset['train'].batch(32)
# test_ds = dataset['test']
#
# # for tmp in train_ds :
# #     images = tmp[0]
# #     labels = tmp[1]
# #
# #     print(images.shape)
# #     print(labels.shape)
# #     break
#
# for images, labels in train_ds :
#     print(images.shape)
#     print(labels.shape)
#     break

# (train_ds, test_ds), ds_info = tfds.load(name='mnist',
#                     shuffle_files=True,
#                     as_supervised=True,
#                     split=['train','test'],
#                     with_info=True)
#
# train_ds = train_ds.batch(32)
#
# for images, labels in train_ds :
#     print(images.shape)
#     print(labels.shape)
#     break

(train_ds, validation_ds, test_ds), ds_info = tfds.load(name='patch_camelyon',
                                                        shuffle_files=True,
                                                        as_supervised=True,
                                                        split=['train','validation','test'],
                                                        with_info=True,
                                                        batch_size=16)

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
images = images.numpy()
labels = labels.numpy()

print(images.shape)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4,4,figsize=(15,15))

for ax_idx, ax in enumerate(axes.flat):
    ax.imshow(images[ax_idx,...])
    ax.set_title(labels[ax_idx],fontsize=30)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()