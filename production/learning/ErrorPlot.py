import pandas as pd
import numpy as np


def get_offset_table_from_two_com_sets(com1,com2):
    '''

    :param com1: a dictionary of COMs of atlas
    :param com2: a dictionary of COMs of beth
    :return:
    '''
    offset_table = pd.DataFrame()
    column_types = ['dx','dy','dz']
    for stack in com2.keys():
        for landmark in com2[stack].keys():
            dx, dy ,dz = np.array(com1[stack][landmark]) - np.array(com2[stack][landmark])
            # highscore = landmark in list(df_highscore[df_highscore['Mouse ID']==stack]['Structure'])
            for data_type in column_types:
                data = {}
                data['landmark'] = landmark
                data['structure'] = landmark+':'+data_type
                data['value'] = eval(data_type)
                data['direction'] = data_type
                data['brain'] = stack
                # data['HighScore'] = highscore
                offset_table = offset_table.append(data, ignore_index=True)
    return offset_table

def create_error_table(tables):
    '''

    :param tables: a pandas table created by get_offset_table_from_two_com_sets
    :return:
    '''
    collection = pd.DataFrame()
    for structure in sorted(set(tables['Structure'])):
        data = {}
        data['Structure'] = structure
        # data['Number of low confidence'] = format(10-len(df_highscore[df_highscore['Structure']==structure]), 'd')
        table = tables[(tables['landmark']==structure) & (tables['Method']=='Detection vs Atlas')]
        for data_type in ['dx','dy','dz']:
            values = np.array(table[table['structure']==structure+':'+data_type]['value'])
            data['Mean of errors: '+data_type] = format(values.mean(),'.2f')
            data['Std of errors: '+data_type] = format(values.std(),'.2f')
            # data['Size of structure: '+data_type] = format(extent[structure+':'+data_type],'.2f')
            data[data_type+': fraction of errors <50'] = '{:.1%}'.format(sum(abs(values)<50)/len(values))
            data[data_type+': fraction of errors [50,100]'] = '{:.1%}'.format(sum((abs(values)<=100) & (abs(values)>=50))/len(values))
            data[data_type+': fraction of errors >100'] = '{:.1%}'.format(sum(abs(values)>100)/len(values))
        collection = collection.append(data,ignore_index=True)[data.keys()]
    return collection

#store your dataframes in a  dict, where the key is the sheet name you want
frames = {'Error wrt atlas': collection, 'Error wrt Beth': collection2}
def save_multiple_table_via_excel(fn,frames):
    writer = pd.ExcelWriter(fn,engine='xlsxwriter')
    for sheet, frame in frames.iteritems():  # .use .items for python 3.X
        frame.to_excel(writer, sheet_name=sheet)

    # critical last step
    writer.save()

