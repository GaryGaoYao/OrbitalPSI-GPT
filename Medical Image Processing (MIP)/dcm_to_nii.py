import SimpleITK as sitk

# 读取dicom文件夹
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("D:\Codes\Skull_Landmarks_OpenAI\Export-Dicom")
reader.SetFileNames(dicom_names)
image = reader.Execute()

# 保存为nii
sitk.WriteImage(image, "D:\Codes\Skull_Landmarks_OpenAI\patient.nii.gz")
