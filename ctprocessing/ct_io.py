#system access modules
import sys, os

#read CT series or .dcm/mhd/raw files
import SimpleITK as sitk
import dicom

#core processing modules
import numpy as np
import cv2


def loadCTSeries():
    """
    Load CT series from Directory using SimpleITK.
    Supported extensions: .dcm, .mhd/raw

    :return numpy array containing images (imageArray)
    """

    from Tkinter import Tk
    from tkFileDialog import askdirectory

    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    dir = askdirectory(initialdir=os.getcwd(),
                       title='Select Exam Directory')  # show an "Open" dialog box and return the path to the selected file

    files = os.listdir(dir)

    # Trasverse directory and look for MHD and RAW file if files quantity is less than 3
    mhd_ok = False
    raw_ok = False
    if len(files) < 10:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".mhd"):
                    mhd_path = root + '/' + file
                    mhd_path = mhd_path.encode('ascii', 'ignore')
                    mhd_ok = True

                if file.endswith(".raw"):
                    raw_ok = True

    # If not found, look for dcm list of files
    if mhd_ok:
        if raw_ok:
            print("Reading MHD and RAW files:", mhd_path)
            img = sitk.ReadImage(mhd_path)
            arrayExam = sitk.GetArrayFromImage(img)
        else:
            print('No RAW file')
            return ''
    else:
        print("Reading Dicom directory:", dir)

        reader = sitk.ImageSeriesReader()

        dicom_path = dir.encode('ascii', 'ignore')
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
        reader.SetFileNames(dicom_names)

        image = reader.Execute()
        arrayExam = sitk.GetArrayFromImage(image)

    return arrayExam;

def loadImages():

    """
    Load DCM or other common image extensions series from list of filenames, using PyDicom.
    Supported extensions: .dcm, .bmp, .jpg, .png

    :return numpy array containing images (imageArray)
    """

    from Tkinter import Tk
    from tkFileDialog import askopenfilenames

    root = Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    fnames = askopenfilenames(title='Select Images', filetypes=[("DICOM files","*.dcm"), ('BMP files', '*.bmp'),
                                                                ('JPG files', '*.jpg'), ('PNG files', '*.png')])


    # Reading Dicom file(s)
    if '.dcm' in fnames[0]:
        # Get ref file
        RefDs = dicom.read_file(fnames[0], force=True)

        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (len(fnames), int(RefDs.Rows), int(RefDs.Columns))

        # Load spacing values (in mm)
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

        # The array is sized based on 'ConstPixelDims'
        imageArray = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        dslist=[]
        # loop through all the DICOM files
        for filenameDCM in fnames:
            # read the file
            dslist.append(dicom.read_file(filenameDCM, force=True))
            # store the raw image data

#TODO: Try to sort series based on DICOM slicelocation tag
        # dslist.sort(key=lambda x: x.SliceLocation, reverse=True)

        for i in range(0,len(dslist)):
            imageArray[i, :, :] = dslist[i].pixel_array


    # Reading other format file(s)
    if '.bmp' in fnames[0] or '.jpg' in fnames[0] or '.png' in fnames[0]:
        # TODO: implement loading of other image extensions (OPENCV?)

        imageArray = np.zeros((len(fnames),512, 512))
        i = 0
        for f in fnames:
            img = cv2.imread(f,cv2.IMREAD_GRAYSCALE)

            imageArray[i, :, :] = img
            i+=1


    return imageArray

def dispImages(imageArray):

    images = np.copy(imageArray)

    LEVEL = -600
    WIDTH = 1500

    def showSlice(x):

        size = images.shape
        if len(size) > 2:
            image = np.array(imageArray[x, :, :], copy=True, dtype=np.float32)
        else:
            image = np.array(imageArray, copy=True, dtype=np.float32)


        if image.max() > 255:


            cv2.imshow('Slices', convert2_8bits(image))
        else:
            cv2.imshow('Slices', image)

    # Create a black image, a window
    cv2.namedWindow('Slices')

    fTime = True
    if  fTime:
        showSlice(0)
        ftime = False

        size = images.shape
        if len(size) > 2:
            cv2.createTrackbar('Index', 'Slices', 0, images.shape[0] - 1, showSlice)

    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

def convert2_8bits(image16b):

    image = np.array(image16b, copy=True, dtype=np.float32)

    if image.max() > 255:
        LEVEL = -600
        WIDTH = 1500
        max = LEVEL + float(WIDTH / 2)
        min = LEVEL - float(WIDTH / 2)
        image.clip(min, max, out=image)
        image -= min
        image /= WIDTH / 255.
        image = image.astype(np.uint8)
    return image

def printArrayExam(arrayExam, path, name=''):

    if arrayExam.max() > 255 or arrayExam.min() < 0:
        LEVEL = -600
        WIDTH = 1500

        max = LEVEL + float(WIDTH / 2)
        min = LEVEL - float(WIDTH / 2)

        if arrayExam.dtype != 'float16':
            arrayExam = arrayExam.astype(np.float16)

        arrayExam.clip(min, max, out=arrayExam)
        arrayExam -= min
        arrayExam /= WIDTH / 255.
        arrayExam = arrayExam.astype(np.uint8)

        print('Writing 16bit Image array')
        print(arrayExam.shape)
        for x in range(0, arrayExam.__len__()):
            file = path + str(x+1) + ".png"
            cv2.imwrite(file, arrayExam[x, :, :])
    else:
        print('Writing Image array')
        print(arrayExam.shape)

        size = arrayExam.shape
        if len(size) > 2:
            for x in range(0, arrayExam.__len__()):
                file = path + str(x+1) + ".png"
                cv2.imwrite(file, arrayExam[x, :, :])
        else:
            file = path + name + ".png"
            cv2.imwrite(file, arrayExam[:, :])

    cv2.waitKey(0)

def renderLung(arrayExam):

    import vtk
    import matplotlib.pyplot as plt
    from vtk.util import numpy_support
    import scipy.io as sio

    # PathDicom = "./Dicom"
    # reader = vtk.vtkDICOMImageReader()
    # reader.SetDirectoryName(PathDicom)
    # reader.Update()

   # x = sio.loadmat('lungContour.mat')
   # lung = x['lung']

    lung2 = np.zeros([512, 512, 512], dtype='uint8')
    lung2[arrayExam > 0]=1
    # plt.imshow(lung2[120, :, :], cmap='gray')
    # plt.show()

    dataImporter = vtk.vtkImageImport()
    data_string = arrayExam.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, 511, 0, 511, 0, 511)
    dataImporter.SetWholeExtent(0, 511, 0, 511, 0, 511)

    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(dataImporter.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(dmc.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor)
    renderer.SetBackground(0, 0, 0)
    renderWindow.Render()
    renderWindowInteractor.Start()


