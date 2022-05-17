import argparse
import os 

class Options():

    def __init__(self):
        self.initialized = False

    def initialize(self,parser):
        #directory
        parser.add_argument('-r', '--root',type=str, help='Root directory for dataset', default='D:\\MDS\\dataset')
        parser.add_argument('-t', '--task',type=str, help='name of MSD Task', default='Task09_Spleen')
        parser.add_argument('-trainfolder', '--trainfolder',type=str, help='folder name which contains training images', default='imagesTr')
        parser.add_argument('-labelfolder', '--labelfolder',type=str, help='folder name which contains training image labels', default='labelsTr')
        parser.add_argument('-type', '--type',type=str, help='File type DICOM or NifTi', default='*.nii')
        parser.add_argument('-split', '--split',type=float, help='File type DICOM or NifTi', default='0.2')
        parser.add_argument('-modeldir', '--modeldir', type=str, help='pathway to pretrained models', default='C:\\Users\\samue\\Documents\\Pro gramming\\MONAI Medical Segmentaiton Decathlon Implementation\\utils\\models')
        parser.add_argument('-config','--config', type=str, help='folder path.', default='C:\\Users\\samue\\Documents\\Pro gramming\\MONAI Medical Segmentaiton Decathlon Implementation\\utils\\config' )
        parser.add_argument('-log','--log', type=str, help='log path.', default='C:\\Users\\samue\\Documents\\Pro gramming\\MONAI Medical Segmentaiton Decathlon Implementation\\utils\\log' )


        #model params
        parser.add_argument('-slice', '--slice', type=int, help='slice number', default='80',)
        #parser.add_argument('--network', default='UNet', help='nnunet, UNet')
        parser.add_argument('-batch_size', type=int, default=4, help='batch size, depends on your machine')
        parser.add_argument('-in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('-out_channels', default=1, type=int, help='Channels of the output')
        parser.add_argument('-epochs', default=250, help='Number of epochs')
        parser.add_argument('-lr', default=0.0001, help='Learning rate')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        # set gpu ids
        return opt