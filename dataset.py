from pathlib import Path

import pydicom
import torch
from pytorch_lightning import LightningDataModule


class IrcadDataloader(LightningDataModule):
    def __init__(self, 
                 dataset_dir:Path, 
                 batch_size:int=1,
                 num_workers:int=0,
                 shuffle:bool=False,
                 drop_last:bool=False
                ):
        super().__init__()
        self.dataset_dir = dataset_dir

        self.labels = []

        # self.ct_scans = {}
        # self.masks = {}
        # self.labels = []
        # self.index_mapping = {}

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def prepare_data(self):
        """
        Download the dataset, unzip the files, etc...
        """
        pass # Do nothing...
    
    def setup(self, stage:str=None) -> None:
        """
        Load the info for each patient, including the labels, masks, and CT scans.
        """
        patients = {}
        for patient_path in self.dataset_dir.iterdir():
            if patient_path.is_dir():
                patient_id = patient_path.name
                patients[patient_id] = {
                    "ct_scans": self.load_ct_scans(patient_path, patient_id),
                    "masks": self.load_masks(patient_path, patient_id)
                }

        # Make sure the labels are sorted.
        self.labels.sort()

        # Split in train, val, and test.
        num_patients = len(patients.keys())
        num_val = max(int(num_patients * 0.1), 1)
        num_test = max(int(num_patients * 0.1), 1)
        num_train = num_patients - (num_val + num_test)

        print(f"Total number of patients: {num_patients}")
        print(f"Number of patients in train set: {num_train}")
        print(f"Number of patients in val set:   {num_val}")
        print(f"Number of patients in test set:  {num_test}")

        val_patients = list(patients.keys())[:num_val]
        test_patients = list(patients.keys())[num_val:num_val+num_test]
        train_patients = list(patients.keys())[num_val+num_test:]

        self.train_ds = Ircadb3D(
            labels=self.labels, 
            ct_scans={patient_id: patients[patient_id]["ct_scans"] for patient_id in train_patients},
            masks={patient_id: patients[patient_id]["masks"] for patient_id in train_patients},
        )
        self.val_ds = Ircadb3D(
            labels=self.labels,
            ct_scans={patient_id: patients[patient_id]["ct_scans"] for patient_id in val_patients},
            masks={patient_id: patients[patient_id]["masks"] for patient_id in val_patients}
        )
        self.test_ds = Ircadb3D(
            labels=self.labels,
            ct_scans={patient_id: patients[patient_id]["ct_scans"] for patient_id in test_patients},
            masks={patient_id: patients[patient_id]["masks"] for patient_id in test_patients}
        )

    def load_ct_scans(self, patient_path:Path, patient_id:str) -> None:
        """
        Load the CT scans for a patient.
        """
        ct_scans = []
        for ct_scan_path in sorted((patient_path / "PATIENT_DICOM").iterdir()):
            if ct_scan_path.is_file():
                ct_scan = pydicom.read_file(ct_scan_path)

                ct_scans.append({
                    "pixel_data": ct_scan.pixel_array,
                    "metadata": {
                        "image_name": ct_scan_path.name,
                        "study_date": ct_scan.StudyDate,
                        "study_time": ct_scan.StudyTime,
                        "gender": ct_scan.PatientSex,
                        "pixel_spacing": ct_scan.PixelSpacing,
                        "slice_thickness": ct_scan.SliceThickness,
                        "image_number": ct_scan.InstanceNumber,
                        "image_position": ct_scan.ImagePositionPatient,
                        "image_orientation": ct_scan.ImageOrientationPatient,
                    }
                })

        return ct_scans


    def load_masks(self, patient_path:Path, patient_id:str) -> dict:
        """
        Load the masks for various organs.
        """
        masks = {}
        for mask_path in (patient_path / "MASKS_DICOM").iterdir():
            organ_name = mask_path.name
            for organ_path in sorted(mask_path.iterdir()):
                if organ_path.is_file():
                    image_name = organ_path.name
                    mask_data = pydicom.read_file(organ_path)

                    # Get the name of the ct scan.
                    if image_name not in masks.keys():
                        masks[image_name] = {}
                    assert organ_name not in masks[image_name], f"Duplicate organ: {organ_name} for {image_name} of patient {patient_id}"

                    masks[image_name][organ_name] = mask_data.pixel_array

            if organ_name not in self.labels:
                self.labels.append(organ_name)

        return masks

    def train_dataloader(self):
        """
        Get the training dataloader.
        """
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )
    
    def val_dataloader(self):
        """
        Get the validation dataloader.
        """
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
    
    def test_dataloader(self):
        """
        Get the test dataloader.
        """
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )


class Ircadb3D(torch.utils.data.Dataset):
    def __init__(self, labels:list, ct_scans:dict, masks:dict) -> None:
        super().__init__()

        self.labels = sorted(labels)
        self.ct_scans = ct_scans
        self.masks = masks

        self.index_mapping = {}
        self.create_index_mapping()
            
    def create_index_mapping(self):
        for patient_id in sorted(self.ct_scans.keys()):
            # Use a mapping to get the correct patient from an index.
            num_scans = len(self.ct_scans[patient_id])
            if len(self.index_mapping) == 0:
                self.index_mapping[num_scans] = patient_id
            else:
                start_idx = sorted(self.index_mapping.keys())[-1]
                end_idx = start_idx + num_scans
                self.index_mapping[end_idx] = patient_id
    
    def __len__(self) -> int:
        # The length of the dataset is the number of ct scans (for each patient).
        length = 0
        for patient_id in self.ct_scans:
            length += len(self.ct_scans[patient_id])

        return length

    def __getitem__(self, index:int) -> dict:
        # Get the patient id and the ct scan index.
        patient_id = None
        for idx in sorted(self.index_mapping.keys()):
            if index < idx:
                patient_id = self.index_mapping[idx]
                break
        
        # Get the internal index of the ct scan for the patient.
        ct_scan_index = (index - idx) + (len(self.ct_scans[patient_id]))
        assert ct_scan_index >= 0 and ct_scan_index < len(self.ct_scans[patient_id]), f"Invalid ct scan index: {ct_scan_index} for patient {patient_id}"

        # Get the metadata for the ct scan.
        metadata = self.ct_scans[patient_id][ct_scan_index]["metadata"]
        image_name = metadata["image_name"]
        
        # Get the ct scan
        ct_scan = torch.tensor(self.ct_scans[patient_id][ct_scan_index]["pixel_data"])
        
        # Get the masks for all available organs.
        masks = []
        for organ_name in self.labels:
            if organ_name in self.masks[patient_id][image_name]:
                # Load the mask for the organ.
                masks.append(torch.tensor(self.masks[patient_id][image_name][organ_name]))
            else:
                # If the mask doesn't exist, create an empty mask.
                masks.append(torch.zeros_like(ct_scan))

        return {
            "ct_scan": ct_scan,
            "masks": masks,
            "metadata": metadata,
        }

if __name__ == "__main__":
    # Test that the dataset loads correctly.
    dataset_dir = Path("/ssd_data/IRCAD/3Dircadb1/")
    dataloader = IrcadDataloader(dataset_dir, num_workers=0, batch_size=1, shuffle=False, drop_last=False)
    dataloader.setup("fit")

    # dataset = dataloader.train_dataloader()
    dataset = dataloader.train_ds
    print(f"Dataset length: {len(dataset)}")

    from pprint import pprint
    print("First item:")
    pprint(dataset[0])
