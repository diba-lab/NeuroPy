import sys
import numpy as np
import pandas as pd
import h5py
import hdf5storage # conda install hdf5storage
from pathlib import Path
from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter


def import_mat_file(mat_import_file='data/RoyMaze1/positionAnalysis.mat', debug_print:bool=True):
    with ProgressMessagePrinter(mat_import_file, action='Loading', contents_description='matlab import file', enable_print=debug_print):
        data = hdf5storage.loadmat(mat_import_file, appendmat=False)
    return data


# ==================================================================================================================== #
# Session XML Related                                                                                                  #
# ==================================================================================================================== #
def find_session_xml(local_session_path, debug_print=False):
    """finds the XML file for a given session provided the session folder path

    Args:
        local_session_path (_type_): _description_

    Usage:
        from neuropy.utils.load_exported import find_session_xml

        # local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15')
        # local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15')
        local_session_path, session_stem, local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-13_14-42-6')

        session_xml_filepath = find_session_xml(local_session_path)
        out_xml_dict, d = LoadXml(session_xml_filepath)
        print(f"active_shank_channels_lists: {out_xml_dict['AnatGrps']}")

    """
    if isinstance(local_session_path, str):
        local_session_path = Path(local_session_path)
    session_stem = local_session_path.stem # '2006-6-08_14-26-15'
    session_xml_filepath = local_session_path.joinpath(session_stem).with_suffix('.xml')
    if debug_print:
        print(f'local_session_path: {local_session_path}')
        print(f'session_xml_filepath: {session_xml_filepath}')
    assert session_xml_filepath.exists() and session_xml_filepath.is_file()
    return session_xml_filepath, session_stem, local_session_path



def LoadXml(session_xml_filepath, MAX_CHANNEL_GROUP_LENGTH = 8, debug_print=False):
    """ Based off of the MATLAB code `load_files\LoadXml.m`
    Usage:
        session_xml_filepath = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.xml')
        assert session_xml_filepath.exists() and session_xml_filepath.is_file()
        out_xml_dict, d = LoadXml(session_xml_filepath)
        out_xml_dict
    """
    # Deal with dataframes:
    def _subfn_buildFinalGroupIdxColumn(df, debug_print=False):
        df['groupIdx'] = list(zip(df.originalGroupIdx , df.splitIdx))
        # uniqueGroupSplitPairs = list(zip(df.originalGroupIdx.to_numpy() , df.splitIdx.to_numpy()))
        if debug_print:
            print(f'uniqueGroupSplitPairs: {df.groupIdx}')
        unique_groupPairs, indices = np.unique(df.groupIdx, return_inverse=True)
        # unique_groupPairs, indices = np.unique(df.groupIdx, return_index=True)
        if debug_print:
            print(f'unique_groupPairs: {unique_groupPairs}, indices: {indices}')
        df['groupIdx'] = indices
        return df

    # Build the initial dictionary datastructure to hold the XML values:
    d = benedict(str(session_xml_filepath.resolve()), format='xml')
    out_xml_dict = {'nBits': int(d['parameters.acquisitionSystem.nBits']), 'nChannels': int(d['parameters.acquisitionSystem.nChannels']), 'samplingRate': int(d['parameters.acquisitionSystem.samplingRate']), 'voltageRange': int(d['parameters.acquisitionSystem.voltageRange']),
 'amplification': int(d['parameters.acquisitionSystem.amplification']), 'offset': int(d['parameters.acquisitionSystem.offset'])}
    out_xml_dict = out_xml_dict | {'Date':d['parameters.generalInfo.date']}
    out_xml_dict = out_xml_dict | {'lfpSampleRate':int(d['parameters.fieldPotentials.lfpSamplingRate'])}
        
    ## Processing Channel Groups:
    out_AnatomicalChannelGroupsList = []
    out_extendedAnatomicalChannelGroupsDataframe = {}
    # Initialize dataframe members:
    for a_key in ['originalGroupIdx','groupIdx','channelIDX','splitIdx']:
        out_extendedAnatomicalChannelGroupsDataframe[a_key] = []

    """    
        [{'channel': [{'@skip': '0', '#text': '72'},
        {'@skip': '0', '#text': '73'},
        {'@skip': '0', '#text': '74'},
        {'@skip': '0', '#text': '75'},
        {'@skip': '0', '#text': '76'},
        {'@skip': '0', '#text': '77'},
        {'@skip': '0', '#text': '78'},
        {'@skip': '0', '#text': '79'}]},
        {'channel': [{'@skip': '0', '#text': '80'},
        {'@skip': '0', '#text': '81'},
        {'@skip': '0', '#text': '82'},
        {'@skip': '0', '#text': '83'},
        {'@skip': '0', '#text': '84'},
        {'@skip': '0', '#text': '85'},
        {'@skip': '0', '#text': '86'},
        {'@skip': '0', '#text': '87'}]},
        ...
        ]
    """
    for i, a_group_dict in enumerate(d['parameters.anatomicalDescription.channelGroups.group']):
        """ a_channel_list: [{'@skip': '0', '#text': '72'}, {'@skip': '0', '#text': '73'}, {'@skip': '0', '#text': '74'}, {'@skip': '0', '#text': '75'}, {'@skip': '0', '#text': '76'}, {'@skip': '0', '#text': '77'}, {'@skip': '0', '#text': '78'}, {'@skip': '0', '#text': '79'}] """
        if debug_print:
            print(a_group_dict['channel'])
        a_channel_list = [int(a_channel_entry['#text']) for a_channel_entry in a_group_dict['channel'] if a_channel_entry['@skip']=='0']
        if debug_print:
            print(a_channel_list)

        out_extendedAnatomicalChannelGroupsDataframe['originalGroupIdx'].extend(np.full_like(a_channel_list, i))
        out_extendedAnatomicalChannelGroupsDataframe['channelIDX'].extend(a_channel_list)

        if debug_print:
            print(a_channel_list)

        # 'groupIdx' must be calculated later, but for now fill with garbage
        out_extendedAnatomicalChannelGroupsDataframe['groupIdx'].extend(np.full_like(a_channel_list, -1))
        if len(a_channel_list) > MAX_CHANNEL_GROUP_LENGTH:
            # split long groups
            if debug_print:
                print(f'MAX_CHANNEL_GROUP_LENGTH: {MAX_CHANNEL_GROUP_LENGTH}')
            chunked_channel_lists = [a_channel_list[i:i+MAX_CHANNEL_GROUP_LENGTH] for i in range(0, len(a_channel_list), MAX_CHANNEL_GROUP_LENGTH)]
            # chunked_channel_lists = [list(array) for array in np.array_split(np.array(a_channel_list), MAX_CHANNEL_GROUP_LENGTH)]
            out_AnatomicalChannelGroupsList.extend(chunked_channel_lists)
            if debug_print:
                print(f'splitting list (original length {len(a_channel_list)}: {a_channel_list} -> {chunked_channel_lists} <num split lists {len(chunked_channel_lists)}>')

            split_idxs = np.array([np.full_like(a_split_list, split_i) for split_i, a_split_list in enumerate(chunked_channel_lists)]).flatten()
            out_extendedAnatomicalChannelGroupsDataframe['splitIdx'].extend(split_idxs)

        else:
            out_AnatomicalChannelGroupsList.append(a_channel_list)
            out_extendedAnatomicalChannelGroupsDataframe['splitIdx'].extend(np.full_like(a_channel_list, 0)) # no split occured

    out_xml_dict['AnatGrps'] = out_AnatomicalChannelGroupsList
    out_xml_dict['nAnatGrps'] = len(out_AnatomicalChannelGroupsList)
    out_xml_dict['AnatGrps_df'] = pd.DataFrame(out_extendedAnatomicalChannelGroupsDataframe)
    # out_xml_dict['AnatGrps_df'] = out_extendedAnatomicalChannelGroupsDataframe
    out_xml_dict['AnatGrps_df'] = _subfn_buildFinalGroupIdxColumn(out_xml_dict['AnatGrps_df'], debug_print=debug_print)
    
    out_xml_dict['ElecGp'] = out_AnatomicalChannelGroupsList
    out_xml_dict['nElecGps'] = len(out_AnatomicalChannelGroupsList)


    ## Processing Spike Groups:
    out_spikeGroupsList = []
    out_extendedSpikeGroupsList = []
    out_extendedSpikeGroupsDataframe = {}
    # Initialize dataframe members:
    for a_key in ['originalGroupIdx','groupIdx','channelIDX','peakSampleIndex','nFeatures','nSamples','splitIdx']:
        out_extendedSpikeGroupsDataframe[a_key] = []

    """
        [{'channels': {'channel': ['0', '1', '2', '3', '4', '5', '6', '7']},
        'nSamples': '54',
        'peakSampleIndex': '26',
        'nFeatures': '3'},
        {'channels': {'channel': ['8', '9', '10', '11', '12', '13', '14', '15']},
        'nSamples': '54',
        'peakSampleIndex': '26',
        'nFeatures': '3'}, ...
    """

    for i, a_group_dict in enumerate(d['parameters.spikeDetection.channelGroups.group']):
        """ a_channel_list: {'channels': {'channel': ['0', '1', '2', '3', '4', '5', '6', '7']}, 'nSamples': '54', 'peakSampleIndex': '26', 'nFeatures': '3'} """
        if debug_print:
            print(a_group_dict)
            print(a_group_dict['channels']['channel']) # ['0', '1', '2', '3', '4', '5', '6', '7']

        a_channel_list = [int(a_channel_entry) for a_channel_entry in a_group_dict['channels']['channel']]
        # build extended properties:
        curr_group_properties = {'spikeDetection_channelGroupIdx': np.full_like(a_channel_list, i),'channels': a_channel_list, 'nSamples': int(a_group_dict['nSamples']), 'peakSampleIndex': int(a_group_dict['peakSampleIndex']), 'nFeatures': int(a_group_dict['nFeatures'])}
        out_extendedSpikeGroupsList.append(curr_group_properties)

        out_extendedSpikeGroupsDataframe['originalGroupIdx'].extend(curr_group_properties['spikeDetection_channelGroupIdx'])
        out_extendedSpikeGroupsDataframe['channelIDX'].extend(curr_group_properties['channels'])

        # Repeat the scalar properties:
        for a_key in ['peakSampleIndex','nFeatures','nSamples']:
            out_extendedSpikeGroupsDataframe[a_key].extend(np.full_like(a_channel_list, curr_group_properties[a_key]))

        if debug_print:
            print(a_channel_list)

        # 'groupIdx' must be calculated later, but for now fill with garbage
        out_extendedSpikeGroupsDataframe['groupIdx'].extend(np.full_like(a_channel_list, -1))

        if len(a_channel_list) > MAX_CHANNEL_GROUP_LENGTH:
            # split long groups
            if debug_print:
                print(f'MAX_CHANNEL_GROUP_LENGTH: {MAX_CHANNEL_GROUP_LENGTH}')
            chunked_channel_lists = [a_channel_list[i:i+MAX_CHANNEL_GROUP_LENGTH] for i in range(0, len(a_channel_list), MAX_CHANNEL_GROUP_LENGTH)]
            out_spikeGroupsList.extend(chunked_channel_lists)
            if debug_print:
                print(f'splitting list (original length {len(a_channel_list)}: {a_channel_list} -> {chunked_channel_lists} <num split lists {len(chunked_channel_lists)}>')
            # Fix df indicies for newly split lists:            
            split_idxs = np.array([np.full_like(a_split_list, split_i) for split_i, a_split_list in enumerate(chunked_channel_lists)]).flatten()
            out_extendedSpikeGroupsDataframe['splitIdx'].extend(split_idxs)

        else:
# groupIdx
            out_extendedSpikeGroupsDataframe['splitIdx'].extend(np.full_like(a_channel_list, 0)) # no split occured
            out_spikeGroupsList.append(a_channel_list)

    out_xml_dict['SpkGrps'] = out_spikeGroupsList
    out_xml_dict['SpkGrps_Extended'] = out_extendedSpikeGroupsList

    

    out_xml_dict['SpkGrps_df'] = pd.DataFrame(out_extendedSpikeGroupsDataframe)
    out_xml_dict['SpkGrps_df'] = _subfn_buildFinalGroupIdxColumn(out_xml_dict['SpkGrps_df'], debug_print=debug_print)


    return out_xml_dict, d

