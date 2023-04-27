
# List of folders & mask number within the data with known issues:
# 110 - 75/218
# Got to 115 - 287/488 (i.e. 506)
# Error on 129... (116/195)

# Error in 122, 127, 128, 129, 149
# Maybe error in 124, 125, 126

import nrrd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime

import pydicom
import torch
from pprint import pprint
from dataset import IrcadDataloader, Ircadb3D
import matplotlib.pyplot as plt


class CTLungDataloader(IrcadDataloader):
    def __init__(self, *args, patient_blacklist=[122, 127, 128, 129, 149, 124, 125, 126], **kwargs):
        """
        Initialization of the CTLungDataLoader which loads in and procceses all of the CT scans.
        """
        super().__init__(*args, **kwargs)
        
        # Blacklist is due to improper masks on truth data
        self.patient_blacklist = patient_blacklist
    
    def setup(self, stage:str=None) -> None:
        """
        Load the info for each patient, including the labels, masks, and CT scans.
        Note each patient potentially has multiple CT scans.
        """
        mask_dir = self.dataset_dir / "OriginalMask"
        image_dir = self.dataset_dir / "OriginalImage"

        patients = {}
        for mask_file in mask_dir.iterdir():
            if mask_file.suffix != ".nrrd":
                continue

            patient_id = mask_file.stem

            # Check the patient id is not on our 'blacklist'
            if int(patient_id) in self.patient_blacklist:
                print(f"Skipping patient {patient_id}.")
                continue

            # Load the masks.
            masks, _ = nrrd.read(mask_file)
            masks = masks.transpose(2, 1, 0) # Swap the first and last axis.

            num_images = len(list(glob.glob(f"{image_dir / patient_id}/*.dcm")))
            patients[patient_id] = {
                "ct_scans": [],
                "masks": []
            }

            # Loop through each mask and retrieve the corresponding CT scan filename.
            for image_num, mask in enumerate(masks):
                # Check the mask is not empty.
                if mask.sum() <= 0:
                    continue

                # Get the CT scan filename.
                ct_scan_path = image_dir / patient_id / f"{num_images - image_num}.dcm"
                if ct_scan_path.exists():
                    patients[patient_id]["ct_scans"].append({
                        "image_name": ct_scan_path.name,
                        "ct_scan_path": ct_scan_path,
                        "pixel_data": [],
                        "mask_path": mask_file,
                        "mask": mask.astype(np.float32),
                        "metadata": {
                            "patient_num": patient_id,
                            "image_num": num_images - image_num,
                            "total_images": num_images,
                            "image_name": ct_scan_path.name,
                        }
                    })

        # Sort the patient id's
        patient_keys = sorted(patients.keys())


        # Split in train, val, and test.
        num_patients = len(patient_keys)
        num_val = max(int(num_patients * 0.1), 1)
        num_test = max(int(num_patients * 0.1), 1)
        num_train = num_patients - (num_val + num_test)

        print(f"Total number of patients: {num_patients}")
        print(f"Number of patients in train set: {num_train}")
        print(f"Number of patients in val set:   {num_val}")
        print(f"Number of patients in test set:  {num_test}")

        val_patients = list(patient_keys)[:num_val]
        test_patients = list(patient_keys)[num_val:num_val+num_test]
        train_patients = list(patient_keys)[num_val+num_test:]

        self.train_ds = CTLung(
            ct_scans={patient_id: patients[patient_id]["ct_scans"] for patient_id in train_patients},
            augment=self.augment,
        )
        self.val_ds = CTLung(
            ct_scans={patient_id: patients[patient_id]["ct_scans"] for patient_id in val_patients},
            augment=False, # Never augment validation data
        )
        self.test_ds = CTLung(
            ct_scans={patient_id: patients[patient_id]["ct_scans"] for patient_id in test_patients},
            augment=False, # Never augment test data
        )

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
            # pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
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
            # pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
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
            persistent_workers=self.persistent_workers,
        )

class CTLung(Ircadb3D):
    def __init__(self, ct_scans:dict, augment:bool=False, verbose:bool=False):
        """
        Initialization of the CTLung which loads procceses a single CT scan.
        """
        super().__init__([], ct_scans, {}, augment, verbose)

    def create_index_mapping(self):
        """
        Creates an easier way to gather data from a specific patient via
        an index value.
        """
        for patient_id in sorted(self.ct_scans.keys()):
            # Use a mapping to get the correct patient from an index.
            num_scans = len(self.ct_scans[patient_id])
            if num_scans > 0:
                if len(self.index_mapping) == 0:
                    self.index_mapping[num_scans] = patient_id
                else:
                    start_idx = sorted(self.index_mapping.keys())[-1]
                    end_idx = start_idx + num_scans
                    self.index_mapping[end_idx] = patient_id

    def __getitem__(self, index: int):
        """
        Get the patient id and the ct scan index.
        """
        patient_id = None
        for idx in sorted(self.index_mapping.keys()):
            if index < idx:
                patient_id = self.index_mapping[idx]
                break
        
        # Get the internal index of the ct scan for the patient.
        ct_scan_index = (index - idx) + len(self.ct_scans[patient_id])
        assert ct_scan_index >= 0 and ct_scan_index < len(self.ct_scans[patient_id]), f"Invalid ct scan index: {ct_scan_index} for patient {patient_id} - (index: {index}, idx: {idx}, len: {len(self.ct_scans[patient_id])})"

        # Load the ct scan from the filesystem (if not already loaded).
        if len(self.ct_scans[patient_id][ct_scan_index]["pixel_data"]) == 0:
            ct_scan_path:Path = self.ct_scans[patient_id][ct_scan_index]["ct_scan_path"]
            ct_scan = pydicom.read_file(ct_scan_path)
            self.ct_scans[patient_id][ct_scan_index]["pixel_data"] = ct_scan.pixel_array

            # Convert the study date to a datetime object.
            study_date = datetime.strptime(ct_scan.StudyDate, "%Y%m%d")
            study_time = datetime.strptime(ct_scan.StudyTime, "%H%M%S.%f")

            # Save the metadata for this ct scan.
            self.ct_scans[patient_id][ct_scan_index]["metadata"].update({
                "image_name": ct_scan_path.name,
                "patient_id": ct_scan.PatientID,
                "study_datetime": datetime.combine(study_date, study_time.time()).isoformat(),
                "gender": ct_scan.PatientSex,
                "pixel_spacing": ct_scan.PixelSpacing,
                "slice_thickness": float(ct_scan.SliceThickness),
                "image_number": int(ct_scan.InstanceNumber),
                "image_position": ct_scan.ImagePositionPatient,
                "image_orientation": ct_scan.ImageOrientationPatient,
            })

        # Get the metadata for the ct scan
        metadata = self.ct_scans[patient_id][ct_scan_index]["metadata"]
        
        # Get the ct scan
        ct_scan = np.array(self.ct_scans[patient_id][ct_scan_index]["pixel_data"]).astype(np.float32)
        ct_scan = torch.tensor(ct_scan)

        # Scale the ct scan to the range [0, 1].
        max_val = ct_scan.max()
        min_val = ct_scan.min()
        ct_scan = (ct_scan - min_val) / (max_val - min_val)
        
        # Get the masks for the lungs.
        mask = np.array(self.ct_scans[patient_id][ct_scan_index]["mask"]).astype(np.float32)
        mask = torch.tensor(mask)

        if self.use_augments:
            ct_scan, mask = self.augment(ct_scan.unsqueeze(0), mask.unsqueeze(0))
            ct_scan = ct_scan.squeeze()
            mask = mask.squeeze()
            
        scan_data = {"ct_scan": ct_scan,
                   "masks": mask,
                   "metadata": metadata}

        return scan_data

if __name__ == "__main__":
    # Test that the dataset loads correctly.
    dataset_dir = Path("/ssd_data/CT_Lung/")
    dataloader = CTLungDataloader(dataset_dir, num_workers=0, batch_size=1, shuffle=False, drop_last=False)
    dataloader.setup("fit")

    # Output general dataset size information    
    dataset = dataloader.train_ds
    print(f"Train Dataset length   : {len(dataset)}")
    print(f"Validate Dataset length: {len(dataloader.val_ds)}")
    print(f"Test Dataset length    : {len(dataloader.test_ds)}")
    print("Index Mapping:")
    pprint(dataset.index_mapping)

    # Sample the first item in the output
    print("First item:")
    batch = dataset[0]
    ct_scan:torch.Tensor = batch["ct_scan"]
    masks:torch.Tensor = batch["masks"]
    metadata = batch["metadata"]
    
    # Print out all of the results
    print(f"CT scan dtype: {ct_scan.dtype}")
    print(f"CT scan shape: {ct_scan.shape}")
    print(f"CT scan max  : {ct_scan.max()}")
    print(f"CT scan min  : {ct_scan.min()}")
    print(f"Masks dtype: {masks.dtype}")
    print(f"Masks shape: {masks.shape}")
    print(f"Masks max  : {masks.max()}")
    print(f"Masks min  : {masks.min()}")
    print("Metadata:")
    pprint(metadata)

    patients_checked = []
    for i, batch in enumerate(dataset):
        print(i, end=', ', flush=True)

        ct_scan:torch.Tensor = batch["ct_scan"]
        masks:torch.Tensor = batch["masks"]
        metadata = batch["metadata"]
        
        # Skipping patients we already looked at
        if metadata['patient_num'] in patients_checked:
            continue

        patients_checked.append(metadata['patient_num'])
        # Show the ct scan and the masks.
        plt.figure(figsize=(10, 10))
        plt.title(f"Patient {metadata['patient_num']} ({metadata['image_num']}/{metadata['total_images']})")
        plt.imshow(ct_scan, cmap="cividis") #viridis, cividis, coolwarm
        plt.imshow(masks * 255, alpha=0.5)
        plt.show()
