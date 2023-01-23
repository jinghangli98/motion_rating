import torch
import nibabel as nib
import glob
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pydicom
import platform
import sys 
import os
study = sys.argv[1]
# study = 'NOV-SCD'
def crop(image, size):
    new_h, new_w = size
    h, w = image.shape
    if h < new_h:
            pad_size = int((new_h - h)/2)
            image = np.pad(image, ((pad_size, pad_size), (0,0)))
    elif h > new_h:
        crop_size = int((h - new_h)/2)
        image = image[crop_size:h-crop_size, :]    
    if w < new_w:
        pad_size = int((new_w - w)/2)
        image = np.pad(image, ((0,0),  (pad_size, pad_size)))
    elif w > new_w:
        crop_size = int((w - new_w)/2)
        image = image[:, crop_size:w-crop_size]
    image = np.stack((image,)*3, axis=0)    
    return image

def print_rating(dcm_path):
    model = torch.load('motion_resnet18_model.pth')
    model.eval()
    
    img_path = glob.glob(f'{dcm_path}/*/*TSE*_ND*/MR*')
    img_path.sort()
    subject_list = [i.split('/')[-3] for i in img_path]
    subject_list = np.unique(subject_list)
    study = dcm_path.split('/')[-2]
    f = open(f"{study}_TSE_rating.txt", "w")    
    for sub in subject_list:
        path = glob.glob(f'{dcm_path}/{sub}/*TSE*/MR*')
        ds = pydicom.filereader.dcmread(path[0])
        _, row, col, _ = ds.AcquisitionMatrix
        img_data = [pydicom.filereader.dcmread(i).pixel_array for i in path]
        img_data = [np.expand_dims(crop(i, (512,512)), 0) for i in img_data]
        img_data = [i/np.max(i) for i in img_data]

        output = [model(torch.tensor(i).float()) for i in img_data]
        output = [int(np.argmax(i.detach().numpy(), axis = 1)) for i in output]
        rating = sum(output)/len(output)
        name = ds.StudyDescription.split('^')[-1]
        name = name + '_' + ds.PatientID
        path = path[0].split('/')[:-1]
        path.insert(1, '/')
        path = os.path.join(*path)
        pdb.set_trace()
        print(f'Coil: {ds.TransmitCoilName} | Patient: {ds.StudyDate}_{name} | Sex: {ds.PatientSex} | DOB: {ds.PatientBirthDate} | MR Sequence: {ds.ProtocolName} | Rating: {rating}')
        f.write(f'Coil: {ds.TransmitCoilName} | Patient: {ds.StudyDate}_{name} | Sex: {ds.PatientSex} | DOB: {ds.PatientBirthDate} | MR Sequence: {ds.ProtocolName} | Rating: {rating} | Path: {path}\n')

if platform.system() == 'Linux':
    dcm_path = f'/run/user/1000/gvfs/sftp:host=136.142.190.89/home/scans/{study}/20*'
elif platform.system() == 'Darwin':
    dcm_path = f'/Volumes/storinator/scans/{study}/20*'
    
print_rating(dcm_path)
    
# model = torch.load('motion_resnet18_model.pth')
# model.eval()
# img_path = glob.glob(f'{dcm_path}/*/*TSE*_ND*/MR*')
# img_path.sort()
# subject_list = [i.split('/')[-3] for i in img_path]
# subject_list = np.unique(subject_list)

# for sub in subject_list:
#     path = glob.glob(f'{dcm_path}/{sub}/*TSE*/MR*')
#     ds = pydicom.filereader.dcmread(path[0])
#     _, row, col, _ = ds.AcquisitionMatrix
    
#     img_data = [pydicom.filereader.dcmread(i).pixel_array for i in path]
#     img_data = [np.expand_dims(crop(i, (512,512)), 0) for i in img_data]
#     img_data = [i/np.max(i) for i in img_data]

#     output = [model(torch.tensor(i).float()) for i in img_data]
#     output = [int(np.argmax(i.detach().numpy(), axis = 1)) for i in output]
#     rating = sum(output)/len(output)
#     name = ds.StudyDescription.split('^')[-1]
#     name = name + '_' + ds.PatientID
#     print(f'Coil: {ds.TransmitCoilName} | Patient: {name} | Sex: {ds.PatientSex} | DOB: {ds.PatientBirthDate} | MR Sequence: {ds.ProtocolName} | Rating: {rating}')

