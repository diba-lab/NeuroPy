from copy import deepcopy
import numpy as np
import panel as pn # for interactive widgets


class DataSessionPanelMixin:
    """ Implementor renders an interactive panel for a session that gives an overview of all of the dataframe-convertable variables for the DataSession object. """
            
    @staticmethod
    def panel_session_dataframes_overview(sess, max_page_items = 20):
        """ Allows a simple overview of a loaded session using Panel.
        Requirements:
            import panel as pn # for interactive widgets
            pn.extension('tabulator') # for dataframes

        Usage:
            panel_session_dataframes_overview(sess)
        """
        def _print_df_shape(df):
            return f'{df.shape[0]} rows x {df.shape[1]} cols'

        # test_columns_dict = {
        #     columns:[
        #         {field:"Photo", title:"photo", formatter:"file"},
        #     ],
        # }
        
        # test_all_kwargs = {
        #  'formatter':'money', 'formatterParams':{'precision':False}   
        # }
        
        # tabs_list = list()
        epochs_df = sess.epochs.to_dataframe().copy()
        epochs_df_widget = pn.Column(pn.widgets.StaticText(name='epochs', value=f'Epochs {_print_df_shape(epochs_df)}'), 
                                    pn.widgets.Tabulator(epochs_df, disabled=True, width_policy='min'))
        # tabs_list.append(('epochs',epochs_df_widget))
        # # laps_df_widget = pn.widgets.Tabulator(sess.laps.to_dataframe())
        # # tabs_list.append(('laps',laps_df_widget))
        pos_df = sess.position.to_dataframe().copy()
        position_df_widget = pn.Column(pn.widgets.StaticText(name='position', value=f'Position {_print_df_shape(pos_df)}'), pn.widgets.Tabulator(pos_df, name='Position', pagination='remote', page_size=max_page_items, disabled=True, width_policy='min'))
        # # tabs_list.append(('position',position_df_widget))
        # # position_df_widget = pn.widgets.Tabulator(sess.position.to_dataframe())
        # # tabs = pn.Tabs(('epochs',epochs_df_widget), ('laps',laps_df_widget), ('position',position_df_widget), dynamic=True)
        # # neurons_plot = plot_raster(sess.neurons, color='jet',add_vert_jitter=True)
        spikes_df = deepcopy(sess.spikes_df)
        #@ workaround to permit display of the Enum-typed 'cell_type' column. Otherwise we get a JSON-encodable error:
        spikes_df.loc[:, 'cell_type'] = np.array([a_type.name for a_type in spikes_df['cell_type'].values])
        spikes_df_widget = pn.Column(pn.widgets.StaticText(name='spikes_df', value=f'Spikes {_print_df_shape(spikes_df)} (flattened spiketrains)'), 
                                    pn.widgets.Tabulator(spikes_df, name='spikes_df', pagination='remote', page_size=max_page_items, hidden_columns=[], disabled=True))

        return pn.Column(pn.Row(epochs_df_widget, pn.Spacer(width=8), position_df_widget), spikes_df_widget, sizing_mode='stretch_width', width_policy='max', min_width=1000)
