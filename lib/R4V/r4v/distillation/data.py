"""
"""
import numpy as np
import os
import re
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets.folder import (
    default_loader,
    find_classes,
    make_dataset,
    IMG_EXTENSIONS,
)


class TeacherStudentDataset(data.Dataset):
    def __init__(self, roots, samples, transforms=None, target_transforms=None):
        self.roots = roots
        self.samples = samples

        self.transforms = transforms
        if self.transforms is not None:
            if not isinstance(self.transforms, list):
                self.transforms = [self.transforms] * len(self.samples)
            assert len(self.transforms) == len(
                self.roots
            ), "must have one transformation per root"
        self.target_transforms = target_transforms
        if self.target_transforms is not None:
            if not isinstance(self.target_transforms, list):
                self.target_transforms = [self.target_transforms] * len(self.samples)
            assert len(self.target_transforms) == len(
                self.roots
            ), "must have one target transformation per root"

    def process_sample(self, sample):
        raise NotImplementedError()

    def assert_same_targets(self, target0, target1):
        assert target0 == target1

    def __getitem__(self, index):
        samples = []
        targets = []
        for i, dataset in enumerate(self.samples):
            sample_, target = dataset[index]
            sample = self.process_sample(sample_)
            if self.transforms is not None:
                sample = self.transforms[i](sample)
            if self.target_transforms is not None:
                target = self.target_transforms[i](target)
            samples.append(sample)
            targets.append(target)
            self.assert_same_targets(target, targets[0])

        return tuple([index] + samples + [target])

    def __len__(self):
        return len(self.samples[0])

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.roots)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, repr(self.transforms).replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, repr(self.target_transforms).replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class DatasetFolders(TeacherStudentDataset):
    def __init__(
        self, *roots, transforms=None, target_transforms=None, loader=default_loader
    ):
        classes_ = []
        class_to_idx_ = []
        samples_ = []
        for root in roots:
            classes, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (
                    RuntimeError(
                        "Found 0 files in subfolders of: " + root + "\n"
                        "Supported extensions are: " + ",".join(IMG_EXTENSIONS)
                    )
                )
            classes_.append(classes)
            class_to_idx_.append(class_to_idx_)
            samples_.append(samples)
            if len(samples_[0]) != len(samples):
                raise ValueError(
                    "Dataset folders must have the same number of samples."
                )
            if len(classes_[0]) != len(classes):
                raise ValueError(
                    "Dataset folders must have the same number of classes."
                )
        super().__init__(roots, samples_, transforms, target_transforms)

        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes_
        self.class_to_idx = class_to_idx_

    def process_sample(self, sample):
        return self.loader(sample)


class UdacityDriving(TeacherStudentDataset):
    def __init__(
        self, *roots, transforms=None, target_transforms=None, loader=default_loader
    ):
        samples_ = []
        for root in roots:
            samples = []
            interpolated_csv = os.path.join(root, "interpolated.csv")
            with open(interpolated_csv) as f:
                _ = f.readline()
                for line in f:
                    split_line = line.split(",")
                    target = np.float32(split_line[6])
                    samples.append((os.path.join(root, split_line[5]), target))
            samples_.append(samples)
            if len(samples_[0]) != len(samples):
                raise ValueError(
                    "Dataset folders must have the same number of samples."
                )

        super().__init__(roots, samples_, transforms, target_transforms)

        self.loader = loader

    def process_sample(self, sample):
        return self.loader(sample)


class Dronet(TeacherStudentDataset):
    def __init__(
        self, *roots, transforms=None, target_transforms=None, loader=default_loader
    ):
        samples_ = []
        for root in roots:
            samples = []
            for subdir in sorted(os.listdir(root)):
                experiment_dir = os.path.join(root, subdir)
                if not os.path.isdir(experiment_dir):
                    continue
                has_steering = os.path.exists(
                    os.path.join(experiment_dir, "sync_steering.txt")
                )
                has_labels = os.path.exists(os.path.join(experiment_dir, "labels.txt"))
                assert has_steering or has_labels, (
                    "Neither steerings nor labels found in %s" % experiment_dir
                )
                assert not (has_steering and has_labels), (
                    "Both steerings and labels found in %s" % experiment_dir
                )
                if has_steering:
                    steering_ground_truth = np.loadtxt(
                        os.path.join(experiment_dir, "sync_steering.txt"),
                        usecols=0,
                        delimiter=",",
                        skiprows=1,
                    )
                    label_ground_truth = np.ones_like(steering_ground_truth) * float(
                        "nan"
                    )
                if has_labels:
                    label_ground_truth = np.loadtxt(
                        os.path.join(experiment_dir, "labels.txt"), usecols=0
                    )
                    steering_ground_truth = np.ones_like(label_ground_truth) * float(
                        "nan"
                    )
                img_dir = os.path.join(experiment_dir, "images")
                files = (
                    name
                    for name in os.listdir(img_dir)
                    if os.path.isfile(os.path.join(img_dir, name))
                    and os.path.splitext(name)[1] in [".png", ".jpg"]
                )
                for frame_number, fname in enumerate(
                    sorted(
                        files, key=lambda fname: int(re.search(r"\d+", fname).group())
                    )
                ):
                    img_path = os.path.join(img_dir, fname)
                    target = np.array(
                        [
                            steering_ground_truth[frame_number],
                            label_ground_truth[frame_number],
                        ],
                        dtype=np.float32,
                    )
                    samples.append((img_path, target))
            samples_.append(samples)
            if len(samples_[0]) != len(samples):
                raise ValueError(
                    "Dataset folders must have the same number of samples."
                )
        super().__init__(roots, samples_, transforms, target_transforms)

        self.loader = loader

    def process_sample(self, sample):
        return self.loader(sample)

    def assert_same_targets(self, target0, target1):
        assert (
            ((target0 == target1) ^ np.isnan(target0))
            & (np.isnan(target0) == np.isnan(target1))
        ).all()

        

class DuplicateTeacher(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        return (index, sample, sample, target)

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return repr(self.dataset)


def get_data_transform(
    transform_config,
    default_height=224,
    default_width=224,
    default_crop_height=None,
    default_crop_width=None,
):
    data_transforms = []
    is_grayscale = transform_config.get("grayscale", False)
    is_bgr = transform_config.get("bgr", False)
    assert not is_grayscale or not is_bgr, "Cannot be both grayscale and bgr"
    if is_grayscale:
        data_transforms.append(transforms.Grayscale(num_output_channels=1))
    data_transforms.append(transforms.ToTensor())
    if is_bgr:
        data_transforms.append(transforms.Lambda(lambda t: t[[2, 1, 0]]))
    if not transform_config.get("presized", True):
        data_transforms.append(transforms.ToPILImage())
        resize_height = transform_config.get("height", default_height)
        resize_width = transform_config.get("width", default_width)
        data_transforms.append(transforms.Resize((resize_height, resize_width)))

        if default_crop_height is not None or default_crop_width is not None:
            default_crop_height = default_crop_height or resize_height
            default_crop_width = default_crop_width or resize_width
            crop_height = transform_config.get("crop_height", default_crop_height)
            crop_width = transform_config.get("crop_width", default_crop_width)
            data_transforms.append(transforms.CenterCrop((crop_height, crop_width)))
        data_transforms.append(transforms.ToTensor())
    data_transforms.append(
        transforms.Lambda(lambda t: t * transform_config.get("max_value", 1.0)) 
    )
    transform_normalize_mean = transform_config.get("mean", None)
    if transform_normalize_mean is not None:
        data_transforms.append(
            transforms.Normalize(
                mean=transform_normalize_mean,
                std=transform_config.get("std", [1.0] * len(transform_normalize_mean)),
            )
        )
    transform = transforms.Compose(data_transforms)
    return transform


def get_data_loader(data_config):
    data_format = data_config.get("format", None).lower()
    if data_format == "cifar10":
        return cifar10(data_config)
    elif data_format == "mnist":
        return mnist(data_config)
    elif data_format == "imagenet":
        return image_net(data_config)
    elif data_format == "udacity-driving":
        return udacity_driving(data_config)
    elif data_format == "dronet":
        return dronet(data_config)
    elif data_format == "acas":
        return acas(data_config)
    else:
        raise ValueError("Unknown data format: %s" % data_format)


def image_net(data_config):
    batch_size = data_config.batch_size

    transform = get_data_transform(
        data_config.transform,
        default_height=256,
        default_width=256,
        default_crop_height=224,
        default_crop_width=224,
    )

    return DataLoader(
        DatasetFolders(
            data_config.teacher.path, data_config.student.path, transforms=transform
        ),
        batch_size=batch_size,
        shuffle=data_config.get("shuffle", False),
        num_workers=1,
        pin_memory=True,
    )


def udacity_driving(data_config):
    batch_size = data_config.batch_size

    transform_config = data_config.transform
    teacher_transform = get_data_transform(
        transform_config.get("teacher", transform_config),
        default_height=100, default_width=100
    )
    student_transform = get_data_transform(
        transform_config.get("student", transform_config),
        default_height=100, default_width=100
    )
    dataset = UdacityDriving(
        data_config.teacher.path, data_config.student.path, transforms=[teacher_transform,student_transform]
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=data_config.get("shuffle", False),
        num_workers=1,
        pin_memory=True,
    )


def mnist(data_config):
    is_train_data = data_config.get("_STAGE", "test") == "train"
    assert data_config.teacher.path == data_config.student.path
    
    transform_config = data_config.transform
    teacher_transform = get_data_transform(
        transform_config.get("teacher", transform_config),
        default_height=28, default_width=28
    )
    student_transform = get_data_transform(
        transform_config.get("student", transform_config),
        default_height=28, default_width=28
    )

    assert data_config.teacher.path == data_config.student.path
    dataset = MNIST(
        data_config.teacher.path,
        transforms=[teacher_transform, student_transform],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=data_config.get("shuffle", False),
        num_workers=1,
    )
    return data_loader


class MNIST(data.Dataset):
    def __init__(self, root, transforms=None, target_transforms=None, loader=default_loader):
        self.dataset = datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)
        self.transforms = transforms
        self.target_transforms = target_transforms

    def assert_same_targets(self, target0, target1):
        assert target0 == target1

    def __getitem__(self, index):
        samples = []
        targets = []

        for i in range(len(self.transforms)):
            sample, target = self.dataset[index]
            if self.transforms is not None:
                sample = self.transforms[i](sample)
            if self.target_transforms is not None:
                target = self.target_transforms[i](target)
            samples.append(sample)
            targets.append(target)
            self.assert_same_targets(target, targets[0])

        return tuple([index] + samples + [target])

    def __len__(self):
        return len(self.dataset)


def acas(data_config):
    batch_size = data_config.batch_size
    dataset = ACAS(
        data_config.teacher.path, data_config.student.path
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=data_config.get("shuffle", False),
        num_workers=1,
        pin_memory=True,
    )


class ACAS(data.Dataset):
    def __init__(self, troot, sroot):
        self.data_teacher = np.load(os.path.join(troot,'data.npy')).astype(np.float32)
        self.labels_teacher = np.load(os.path.join(troot,'label.npy')).astype(np.float32)
        self.data_student = np.load(os.path.join(troot,'data.npy')).astype(np.float32)
        self.labels_student = np.load(os.path.join(troot,'label.npy')).astype(np.float32)
        
    def __getitem__(self, index):
        sample_data_teacher = torch.from_numpy(self.data_teacher[index])
        sample_labels_teacher = torch.from_numpy(self.labels_teacher[index])
        sample_data_student = torch.from_numpy(self.data_student[index])
        sample_labels_student = torch.from_numpy(self.labels_student[index])
        
        assert torch.equal(sample_labels_teacher, sample_labels_student)
        return index,sample_data_teacher,sample_data_student,sample_labels_teacher
    

    def __len__(self):
        return len(self.data_teacher)


def dronet(data_config):
    batch_size = data_config.batch_size

    transform_config = data_config.transform

    teacher_transform = get_data_transform(
        transform_config.get("teacher", transform_config),
        default_height=240,
        default_width=320,
        default_crop_height=200,
        default_crop_width=200,
    )
    student_transform = get_data_transform(
        transform_config.get("student", transform_config),
        default_height=240,
        default_width=320,
        default_crop_height=200,
        default_crop_width=200,
    )

    dataset = Dronet(
        data_config.teacher.path,
        data_config.student.path,
        transforms=[teacher_transform, student_transform],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=data_config.get("shuffle", False),
        num_workers=1,
        pin_memory=True,
    )


class CIFAR10:
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        if self.train:
            train_data_path = os.path.join(root, "train_data.npy")
            train_labels_path = os.path.join(root, "train_labels.npy")
            self.train_data = np.load(train_data_path)
            self.train_data = torch.from_numpy(self.train_data.astype("float32"))
            self.train_labels = np.load(train_labels_path).astype("int")
        else:
            test_data_path = os.path.join(root, "test_data.npy")
            test_labels_path = os.path.join(root, "test_labels.npy")
            self.test_data = np.load(test_data_path)
            self.test_data = torch.from_numpy(self.test_data.astype("float32"))
            self.test_labels = np.load(test_labels_path).astype("int")

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target


def cifar10(data_config):
    is_train_data = data_config.get("_STAGE", "test") == "train"
    assert data_config.teacher.path == data_config.student.path

    data_set = CIFAR10(root=data_config.teacher.path, train=is_train_data)

    data_loader = DataLoader(
        DuplicateTeacher(data_set),
        batch_size=data_config.batch_size,
        shuffle=data_config.get("shuffle", False),
        num_workers=1,
    )
    return data_loader
