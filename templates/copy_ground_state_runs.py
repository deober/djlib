import os, sys

"""
This is a standalone function that'll copy all the calculations for specific configs (ie. ground states) from a CASm project to the a subfolder in the current directory
"""

root_dir = os.getcwd() #running from the "ground_states" folder
#structure_file = 'structure.json'
fcc_data_dir = os.path.join(root_dir, '../../1.1.2_HfN_FCC_octa_redux')
hcp_data_dir = os.path.join(root_dir, '../../1.1.2_HfN_HCP_octa_redux')
fcc_list = os.path.join(root_dir, 'FCC_gs.txt') #format is config on each line ex: SCEL1_1_1_1_0_0_0/1
hcp_list = os.path.join(root_dir, 'HCP_gs.txt')

'''
#FCC
for config in open(fcc_list):
    os.system('cp ' + os.path.join(fcc_data_dir,'training_data',config.strip(),structure_file) + ' ' + root_dir + '/FCC/' + config.strip().replace("/","-"))
    
#HCP
for config in open(hcp_list):
    os.system('cp ' + os.path.join(hcp_data_dir,'training_data',config.strip(),structure_file) + ' ' + root_dir + '/HCP/' + config.strip().replace("/","-"))
'''

## copy whole folder of vasp runs for those configs
#FCC
for config in open(fcc_list):
    os.system('mkdir -p ' + root_dir + '/FCC/' + config.strip().split('/')[0])
    os.system('cp -r ' + os.path.join(fcc_data_dir,'training_data',config.strip()) + '* ' + root_dir + '/FCC/' + config.strip())
    
#HCP
for config in open(hcp_list):
    os.system('mkdir -p ' + root_dir + '/HCP/' + config.strip().split('/')[0])
    os.system('cp -r ' + os.path.join(hcp_data_dir,'training_data',config.strip()) + '* ' + root_dir + '/HCP/' + config.strip())