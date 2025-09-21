import multiprocessing

from torchvision import transforms
from torch.utils import data

from data.dataset_duck import SegDataset_out


def get_dataloaders_output(input_paths):

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_dataset = SegDataset_out(
        input_paths=input_paths,
        transform_input=transform_input4test,
    )
    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    return test_dataloader
