from __future__ import unicode_literals, print_function
from builtins import str
from distutils.spawn import find_executable
from  AD_Dataset import  Dataset_Import as ad_dataset


def read_directory_file(self, original_dir, source=None):
    group_data = []
    try:
        if original_dir is not None:

            for file in os.listdir(original_dir):

                if os.path.isdir(os.path.join(original_dir, file)):
                    filepath = os.path.join(original_dir, file)

                    for sub_file in os.listdir(filepath):

                        if os.path.isdir(os.path.join(filepath, sub_file)):

                            filepath2 = os.path.join(filepath, sub_file)

                            for sub_file2 in os.listdir(filepath2):
                                decision_check = None

                                if self.strict_match:
                                    decision_check = sub_file2.strip() in self.image_group

                                    # str(self.image_group) == str(sub_file.strip())
                                if decision_check == True:

                                    modal_groupings_source = os.path.join(filepath2, sub_file2)

                                    files_in_time = len(os.listdir(modal_groupings_source))
                                    if files_in_time > 1:
                                        time_file = max(os.listdir(modal_groupings_source))
                                    else:
                                        time_file = os.listdir(modal_groupings_source)[0]

                                    time_grouping_source = os.path.join(modal_groupings_source, time_file)
                                    files_in_time = len(os.listdir(modal_groupings_source))

                                    for image_file in os.listdir(time_grouping_source):

                                        image_grouping_source = os.path.join(time_grouping_source, image_file)

                                        if os.path.isdir(os.path.join(image_grouping_source)):

                                            for image_file in os.listdir(image_grouping_source):
                                                image_grouping_source_file = os.path.join(image_grouping_source,
                                                                                          image_file)
                                                print(image_grouping_source,end="\n")



    except OSError as e:
        print('Error: %s' % e)

    return group_data


data=ad_dataset()

read_directory_file(data.train_ad_dir)

exit(1)
print('Message to user:')

if(find_executable('bse') and find_executable('svreg.sh') and find_executable('bdp.sh')):
    print('Your system path has been set up correctly. Continue on with the tutorial.')
else:
    print('Your system path has not been set up correctly.')
    print('Please add the above paths to your system path variable and restart the kernel for this tutorial.')
    print('Edit your ~/.bashrc file, and add the following line, replacing your_path with the path to BrainSuite16a1:\n')
    print('export PATH=$PATH:/your_path/BrainSuite16a1/svreg/bin:/your_path/BrainSuite16a1/bdp:/your_path/BrainSuite16a1/bin')
    exit(1)

#Path set properly if reached here
from nipype import config #Set configuration before importing nipype pipeline
cfg = dict(execution={'remove_unnecessary_outputs' : False}) #We do not want nipype to remove unnecessary outputs
config.update_config(cfg)

import nipype.pipeline.engine as pe
import nipype.interfaces.brainsuite as bs
import nipype.interfaces.io as io
import os


from distutils.spawn import find_executable
brainsuite_atlas_directory ="/usr/local/BrainSuite18a/atlas/"
#find_executable('bse')[:-3] + '../atlas/'



brainsuite_workflow = pe.Workflow(name='brainsuite_workflow_cse')
brainsuite_workflow.base_dir='./'


bseObj = pe.Node(interface=bs.Bse(), name='BSE')
bseObj.inputs.inputMRIFile = os.path.join(os.getcwd(),"BrainSuiteNipype_Tutorial/2523412.nii.gz")
bfcObj = pe.Node(interface=bs.Bfc(),name='BFC')
pvcObj = pe.Node(interface=bs.Pvc(), name = 'PVC')
cerebroObj = pe.Node(interface=bs.Cerebro(), name='CEREBRO')
#Provided atlas files
cerebroObj.inputs.inputAtlasMRIFile =(brainsuite_atlas_directory + 'brainsuite.icbm452.lpi.v08a.img')
cerebroObj.inputs.inputAtlasLabelFile = (brainsuite_atlas_directory + 'brainsuite.icbm452.v15a.label.img')
cortexObj = pe.Node(interface=bs.Cortex(), name='CORTEX')
scrubmaskObj = pe.Node(interface=bs.Scrubmask(), name='SCRUBMASK')
tcaObj = pe.Node(interface=bs.Tca(), name='TCA')
dewispObj=pe.Node(interface=bs.Dewisp(), name='DEWISP')
dfsObj=pe.Node(interface=bs.Dfs(),name='DFS')
pialmeshObj=pe.Node(interface=bs.Pialmesh(),name='PIALMESH')
hemisplitObj=pe.Node(interface=bs.Hemisplit(),name='HEMISPLIT')

brainsuite_workflow.add_nodes([bseObj, bfcObj, pvcObj, cerebroObj, cortexObj, scrubmaskObj, tcaObj, dewispObj, dfsObj, pialmeshObj, hemisplitObj])

brainsuite_workflow.connect(bseObj, 'outputMRIVolume', bfcObj, 'inputMRIFile')
brainsuite_workflow.connect(bfcObj, 'outputMRIVolume', pvcObj, 'inputMRIFile')
brainsuite_workflow.connect(bfcObj, 'outputMRIVolume', cerebroObj, 'inputMRIFile')
brainsuite_workflow.connect(cerebroObj, 'outputLabelVolumeFile', cortexObj, 'inputHemisphereLabelFile')
brainsuite_workflow.connect(pvcObj, 'outputTissueFractionFile', cortexObj, 'inputTissueFractionFile')
brainsuite_workflow.connect(cortexObj, 'outputCerebrumMask', scrubmaskObj, 'inputMaskFile')
brainsuite_workflow.connect(cortexObj, 'outputCerebrumMask', tcaObj, 'inputMaskFile')
brainsuite_workflow.connect(tcaObj, 'outputMaskFile', dewispObj, 'inputMaskFile')
brainsuite_workflow.connect(dewispObj, 'outputMaskFile', dfsObj, 'inputVolumeFile')
brainsuite_workflow.connect(dfsObj, 'outputSurfaceFile', pialmeshObj, 'inputSurfaceFile')
brainsuite_workflow.connect(pvcObj, 'outputTissueFractionFile', pialmeshObj, 'inputTissueFractionFile')
brainsuite_workflow.connect(cerebroObj, 'outputCerebrumMaskFile', pialmeshObj, 'inputMaskFile')
brainsuite_workflow.connect(dfsObj, 'outputSurfaceFile', hemisplitObj, 'inputSurfaceFile')
brainsuite_workflow.connect(cerebroObj, 'outputLabelVolumeFile', hemisplitObj, 'inputHemisphereLabelFile')
brainsuite_workflow.connect(pialmeshObj, 'outputSurfaceFile', hemisplitObj, 'pialSurfaceFile')

brainsuite_workflow.run()

#Print message when all processing is complete.
print('Processing has completed.')