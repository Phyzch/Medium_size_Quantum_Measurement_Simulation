import os
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.gridspec as gridspec

def Read_Energy_list(file_path):
    with open(file_path) as f:
        data = f.read().splitlines()
        datalen = len(data)

        parameter_num = int( (datalen - 1) / 2 )

        line = data[0].strip('\n')
        line = re.split(' ', line)
        wave_func = [ float(i) for i in line if i!='']

        line_index = 2
        localization_energy_list = []
        for i in range(parameter_num):
            line = data[line_index]
            line = re.split(' ', line)
            localization_energy = float(line[0])

            localization_energy_list.append(localization_energy)

            line_index = line_index + 2

        return localization_energy_list , wave_func

def Read_left_right_energy_list(folder_path):
    left_file_path = os.path.join(folder_path, "localization_parameter_list_left.txt")
    right_file_path = os.path.join(folder_path , "localization_parameter_list_right.txt")

    left_localization_energy_list, wave_func = Read_Energy_list(left_file_path)
    right_localization_energy_list , wave_func = Read_Energy_list(right_file_path)

    return left_localization_energy_list , right_localization_energy_list , wave_func

def Analyze_left_right_localization_dist(folder_path):
    left_localization_energy_list, right_localization_energy_list, wave_func = Read_left_right_energy_list(folder_path)

    theoretical_localization_prob = [ pow(i,2) for i in wave_func]

    block_num = 50

    fig = plt.figure(figsize=(10, 20))
    spec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
    spec.update(hspace=0.5, wspace=0.3)
    ax_1 = fig.add_subplot(spec[0, 0])
    ax_2 = fig.add_subplot(spec[0, 1])

    ax_1.hist(left_localization_energy_list, bins = block_num, range = (0,1) , density = True)
    ax_1.set_ylabel('prob dist. ')
    ax_1.set_title('left side.   Localization prob =   ' + str( round(theoretical_localization_prob[0] , 3) )  )

    ax_2.hist(right_localization_energy_list, bins = block_num, range = (0,1) , density = True)
    ax_2.set_ylabel('prob dist. ')
    ax_2.set_title('right side. Localization prob =  ' + str(round( theoretical_localization_prob[1] , 3 )))

    plt.show()



